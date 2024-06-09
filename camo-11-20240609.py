import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter

# Function to generate initial conditions
def generate_initial_conditions(size):
    pattern = np.random.rand(size, size, 3)
    return pattern

# Function to make the array/image seamless
def make_seamless(array):
    size = array.shape[0]
    seamless = np.copy(array)

    # Wrap around the edges
    seamless[:size//2, :size//2] = (array[:size//2, :size//2] + np.roll(array, shift=-(size//2), axis=(0,1))[:size//2, :size//2]) / 2
    seamless[:size//2, size//2:] = (array[:size//2, size//2:] + np.roll(array, shift=-(size//2), axis=(0,1))[:size//2, size//2:]) / 2
    seamless[size//2:, :size//2] = (array[size//2:, :size//2] + np.roll(array, shift=(size//2), axis=(0,1))[size//2:, :size//2]) / 2
    seamless[size//2:, size//2:] = (array[size//2:, size//2:] + np.roll(array, shift=(size//2), axis=(0,1))[size//2:, size//2:]) / 2

    # Blend edges
    seamless[:size//10, :] = (seamless[:size//10, :] + np.roll(seamless, shift=size//2, axis=0)[:size//10, :]) / 2
    seamless[-size//10:, :] = (seamless[-size//10:, :] + np.roll(seamless, shift=-size//2, axis=0)[-size//10:, :]) / 2
    seamless[:, :size//10] = (seamless[:, :size//10] + np.roll(seamless, shift=size//2, axis=1)[:, :size//10]) / 2
    seamless[:, -size//10:] = (seamless[:, -size//10:] + np.roll(seamless, shift=-size//2, axis=1)[:, -size//10:]) / 2

    return seamless

# Function to apply transformations to a single channel
def transform_channel(channel, iterations, sigma_x, sigma_y, brightness):
    for _ in range(iterations):
        channel = gaussian_filter(channel, sigma=(sigma_y, sigma_x))
        pil_image = Image.fromarray((channel * 255).astype(np.uint8), mode='L')
        pil_image = pil_image.filter(ImageFilter.SHARPEN)
        channel = np.array(pil_image) / 255.0
        channel = make_seamless(channel)
    # Apply brightness
    channel *= brightness
    return channel

# Function to apply transformations to the pattern
def transform_pattern(pattern, iterations, sigma_x, sigma_y, brightness):
    channels = []
    for i in range(3):
        channel = pattern[:, :, i]
        transformed_channel = transform_channel(channel, iterations[i], sigma_x[i], sigma_y[i], brightness[i])
        channels.append(transformed_channel)
    return np.stack(channels, axis=-1)

# Function to scale the pattern
def scale_pattern(pattern, scale_factor=10):
    pattern = np.kron(pattern, np.ones((scale_factor, scale_factor, 1)))
    return pattern

# Function to overlay patterns on a background image
def overlay_patterns_on_background(background_image_path, patterns, positions):
    background = Image.open(background_image_path).convert('RGB')
    background = np.array(background)
    
    for pattern, position in zip(patterns, positions):
        x, y = position
        pattern_size = pattern.shape[0]
        pattern_image = (pattern * 255).astype(np.uint8)
        
        # Overlay pattern on the background
        background[y:y+pattern_size, x:x+pattern_size] = pattern_image
    
    return Image.fromarray(background, mode='RGB')

# Display image for user selection in fullscreen
def display_selection_image(image, positions, size):
    fig, ax = plt.subplots(figsize=(20, 20))  # Larger figure size
    ax.imshow(image)
    
    # Create invisible rectangles for selection
    rects = []
    for x, y in positions:
        rect = patches.Rectangle((x, y), size, size, linewidth=1, edgecolor='none', facecolor='none')
        ax.add_patch(rect)
        rects.append(rect)
    
    selected_indices = []

    def on_click(event):
        for i, rect in enumerate(rects):
            if rect.contains_point((event.x, event.y)):
                print(f"Selected pattern: {i}")
                selected_indices.append(i)
                rect.set_edgecolor('g')
                fig.canvas.draw()
                if len(selected_indices) >= 4:  # Allow selection of 4 patterns
                    plt.close(fig)
                break
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=True)
    return selected_indices

# Function to crossover two genomes
def crossover(genome1, genome2):
    new_genome = {}
    for key in genome1:
        if isinstance(genome1[key], list):
            new_genome[key] = [random.choice([genome1[key][i], genome2[key][i]]) for i in range(3)]
        else:
            new_genome[key] = random.choice([genome1[key], genome2[key]])
    return new_genome

# Function to mutate a genome
def mutate(genome, mutation_rate):
    new_genome = genome.copy()
    if random.random() < mutation_rate:
        new_genome['pattern'] = np.clip(new_genome['pattern']+mutation_rate*np.random.rand(size, size, 3),0,1)
    if random.random() < mutation_rate:
        new_genome['iterations'] = [np.clip(new_genome['iterations'][i] + np.random.randint(-2, 3), 0, 20) for i in range(3)]
    if random.random() < mutation_rate:
        new_genome['sigma_x'] = [np.clip(new_genome['sigma_x'][i] + mutation_rate*np.random.uniform(-0.3, 0.3), 0, 2) for i in range(3)]
    if random.random() < mutation_rate:
        new_genome['sigma_y'] = [np.clip(new_genome['sigma_y'][i] + mutation_rate*np.random.uniform(-0.3, 0.3), 0, 2) for i in range(3)]
    if random.random() < mutation_rate:
        new_genome['brightness'] = [np.clip(new_genome['brightness'][i] + mutation_rate*np.random.uniform(-0.3, 0.3), 0, 1) for i in range(3)]
    return new_genome

# Function to generate the next generation
def next_generation(best_genomes, generation_num, folder_path):
    next_gen_genomes = []
    while len(next_gen_genomes) < 4:
        parent1, parent2 = random.sample(best_genomes, 2)
        child_genome = mutate(crossover(parent1, parent2),1.0/float(generation_num))
        next_gen_genomes.append(child_genome)
    
    # Save new generation
    new_folder_path = os.path.join(folder_path, f'generation_{generation_num}')
    os.makedirs(new_folder_path, exist_ok=True)
    
    for i, genome in enumerate(next_gen_genomes):
        pattern = generate_initial_conditions(size)
        transformed_pattern = transform_pattern(pattern, genome['iterations'], genome['sigma_x'], genome['sigma_y'], genome['brightness'])
        scaled_pattern = scale_pattern(transformed_pattern)
        
        # Ensure the scaled pattern is in the 0..1 range
        scaled_pattern = np.clip(scaled_pattern, 0, 1)
        
        plt.imsave(os.path.join(new_folder_path, f'pattern_{i}.png'), scaled_pattern)
    
    return next_gen_genomes

# Main program
individuals=16
kill=4
keep=individuals-kill
size = 30
background_image_path = 'flecky-camo-pattern.jpg'  # Replace with your image path
folder_path = './generations'

# Initialize genomes
initial_genomes = [
    {
     'pattern': generate_initial_conditions(size),
     'iterations': [random.randint(0, 20), random.randint(0, 20), random.randint(0, 20)],
     'sigma_x': [random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)],
     'sigma_y': [random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)],
     'brightness': [random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)]
     }
     for _ in range(individuals)
]

# Load the background image to get its dimensions
background = Image.open(background_image_path)
bg_width, bg_height = background.size

# Repeat process for multiple generations
num_generations = 20
for generation_num in range(1, num_generations + 1):
    # Generate initial patterns and overlay them
    patterns = [scale_pattern(transform_pattern(genome['pattern'], genome['iterations'], genome['sigma_x'], genome['sigma_y'], genome['brightness'])) for genome in initial_genomes]

    # Generate random positions for the initial overlay
    positions = [
        (random.randint(0, bg_width - size * 10), random.randint(0, bg_height - size * 10))
        for _ in range(individuals)
    ]

    overlayed_image = overlay_patterns_on_background(background_image_path, patterns, positions)

    # Display for selection
    selected_indices = display_selection_image(overlayed_image, positions, size * 10)
    selected_genomes = [initial_genomes[i] for i in selected_indices]
    unselected_genomes = [initial_genomes[i] for i, _ in enumerate(initial_genomes) if i not in selected_indices]

    # Generate next generation
    next_gen_genomes = next_generation(unselected_genomes, generation_num, folder_path)

    # Store the best genomes for the next iteration
    initial_genomes = unselected_genomes + random.sample(next_gen_genomes, kill)

