import os

def delete_all_files_in_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file before attempting to delete
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")

# Example usage:
# folder_path = 'problems-2'
# delete_all_files_in_folder(folder_path)
folder_path = 'results-3'
delete_all_files_in_folder(folder_path)
