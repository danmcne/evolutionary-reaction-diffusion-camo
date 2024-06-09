# Camo Pattern Evolution

This project generates evolving camouflage patterns using genetic algorithms. Each generation of patterns is transformed using a set of parameters (iterations, sigma values for Gaussian blur, and brightness) that mutate and crossover to produce new generations. The user can select the best patterns from each generation, which then serve as the basis for the next generation.

## Features

- **Pattern Generation**: Generates initial random patterns.
- **Seamless Transformation**: Ensures patterns are seamless for continuous tiling.
- **Channel-wise Transformation**: Applies transformations separately to each color channel.
- **Interactive Selection**: Allows users to select the best patterns from each generation.
- **Genetic Algorithm**: Uses mutation and crossover to evolve patterns over multiple generations.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Pillow

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/evolutionary-reaction-diffusion-camo.git
    cd evolutionary-reaction-diffusion-camo
    ```

2. Install the required libraries:
    ```sh
    pip install numpy scipy matplotlib pillow
    ```

3. Place your background image in the project directory and update the `background_image_path` variable in the script with its name.

4. Run the script:
    ```sh
    python camo_pattern_evolution.py
    ```

5. A window will open displaying the generated patterns on the background image. Click on the patterns you want to **remove** from the next generation.

6. The **unselected** patterns will be used to generate the next generation of patterns.

## Parameters

Each genome consists of the following parameters:

- `iterations`: Number of iterations for the transformation.
- `sigma_x`: Sigma value for the Gaussian blur in the x-direction.
- `sigma_y`: Sigma value for the Gaussian blur in the y-direction.
- `brightness`: Brightness multiplier for each channel.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Author

ihaphleas

## Acknowledgements

Inspired by the principles of genetic algorithms and pattern evolution.
