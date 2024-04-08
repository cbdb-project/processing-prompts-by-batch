# Processing Prompts by Batch (probat)

Processing Prompts by Batch is a Python script designed to automate the process of sending batches of prompts to the [G4F](https://github.com/xtekky/gpt4free) API (an illustrative API representing services similar to OpenAI's GPT-3.5 models. It's FREE and LEGAL!) and collecting their responses. It reads prompts from a text file, sends them to the G4F API, and saves the processed output in another text file. The script is useful for bulk processing of text data, leveraging the G4F API for text generation or transformation tasks.

## Features

- **Batch Processing:** Process multiple prompts in batches, reducing the overhead of sending individual requests.
- **Customizable Timing:** Configurable timeouts between API calls to prevent rate limiting.
- **Text Cleaning:** Cleans up the output text by replacing newline characters with HTML line breaks for easier web display.
- **Dynamic Batch Sizing:** Adjusts the batch size based on the number of prompts to ensure efficient processing.
- **Output Management:** Automatically creates an output file for the processed prompts and ensures no duplication by removing any existing output file at the start.

## Requirements

- Python 3.6 or newer
- `g4f` Python package

## Installation

Before running the script, ensure you have Python installed on your system. You can then install the required `g4f` package using pip:

```bash
pip install -U g4f
```

## Usage

1. Place your text prompts in a file named `prompts.txt`, with one prompt per line.
2. Run probat.py.
3. The script will process all prompts and save the outputs in `output.txt`.

## Configuration

You can adjust the following configurations at the beginning of the script:

- `TEMP_BATCH_SIZE`: The number of prompts to process in each batch (default: 10).
- `TIMEOUT`: Base timeout in seconds between batches (default: 0.5 seconds).
- `TIMEOUT_OFFSET`: Additional random timeout offset to prevent consistent timing patterns (default: 0.5).

## Limitations

- The script is designed to work with the `g4f` package and the G4F API. Ensure compatibility and proper installation before use.
- The batch size and timeouts may need adjustment based on the API's rate limits and your specific requirements.

## Disclaimer

This script is provided as-is, and it is a generic template for interacting with text generation APIs. Always adhere to the API provider's usage policies and guidelines.

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
