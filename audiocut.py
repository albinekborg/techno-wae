from pydub import AudioSegment
import os

def split_mp3(input_file, output_dir, chunk_length=15):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'rb') as mp3_file:  # Open in binary mode
        audio = AudioSegment.from_mp3(mp3_file)
    
    total_length = len(audio)

    chunk_start = 0
    chunk_number = 1

    while chunk_start < total_length:
        chunk_end = chunk_start + chunk_length * 1000  # Convert to milliseconds
        if chunk_end > total_length:
            chunk_end = total_length

        chunk = audio[chunk_start:chunk_end]
        output_file = os.path.join(output_dir, f'chunk_{chunk_number}.mp3')
        chunk.export(output_file, format="mp3")

        chunk_start = chunk_end
        chunk_number += 1

if __name__ == "__main__":
    input_file = "./music_src/lofi_deep_house.mp3"  # Replace with the path to your input MP3 file
    output_dir = "output_chunks"  # Replace with the directory where you want to save the chunks
    chunk_length = 15  # Length of each chunk in seconds

    split_mp3(input_file, output_dir, chunk_length)
