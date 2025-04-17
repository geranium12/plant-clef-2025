import os
import shutil
from concurrent.futures import ThreadPoolExecutor


def copy_file(src_file_path: str, dst_file_path: str) -> None:
    if os.path.exists(dst_file_path):
        print(f"File already exists: {dst_file_path}")
    else:
        shutil.copy2(src_file_path, dst_file_path)


if __name__ == "__main__":
    src_dir: str = ""
    dst_dir: str = ""
    os.makedirs(dst_dir, exist_ok=True)

    print(f"Copying files from {src_dir} to {dst_dir}...")

    with ThreadPoolExecutor(max_workers=256) as executor:
        with os.scandir(src_dir) as folders:
            for folder in folders:
                if not folder.is_dir():
                    continue
                dst_folder_path = os.path.join(dst_dir, folder.name)
                os.makedirs(dst_folder_path, exist_ok=True)
                with os.scandir(folder.path) as files:
                    for file in files:
                        if not file.is_file():
                            continue
                        src_file_path = file.path
                        dst_file_path = os.path.join(dst_folder_path, file.name)
                        executor.submit(copy_file, src_file_path, dst_file_path)
