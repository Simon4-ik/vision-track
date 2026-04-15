import os
import shutil
from pathlib import Path
from ultralytics.utils.downloads import download

def main():
    target_dir = Path("data/coco_dataset")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download coco8.zip
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco8.zip"
    download([url], dir=str(target_dir))
    
    # After download, ultralytics extracts the contents into target_dir / "coco8"
    extracted_dir = target_dir / "coco8"
    if extracted_dir.exists():
        # Move all contents from target_dir/coco8 to target_dir
        for item in extracted_dir.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        extracted_dir.rmdir()
        
    print(f"Dataset ready at: {target_dir}")
    
    # fix paths in data.yaml if it contains absolute paths
    data_yaml = target_dir / "data.yaml"
    if data_yaml.exists():
        content = data_yaml.read_text(encoding="utf-8")
        # replace any path setting
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('path:'):
                lines[i] = f"path: {target_dir.absolute()}"
        data_yaml.write_text('\n'.join(lines), encoding="utf-8")
        print("Updated data.yaml")

if __name__ == "__main__":
    main()
