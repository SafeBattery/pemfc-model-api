import subprocess

required_packages = [
    "flask",
    "torch",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "requests",
    "pandas"
]

with open("requirements.txt", "w", encoding="utf-8") as f:
    for pkg in required_packages:
        try:
            result = subprocess.run(
                ["pip", "show", pkg],
                capture_output=True,
                text=True,
                encoding="utf-8",  # <-- 이 부분 추가!
                check=True
            )
            version_line = None
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        version = parts[1].strip()
                        version_line = f"{pkg}=={version}"
                        break
            if version_line:
                f.write(version_line + "\n")
            else:
                print(f"[WARN] Version not found for {pkg}")
        except subprocess.CalledProcessError:
            print(f"[ERROR] {pkg} is not installed!")
        except UnicodeDecodeError as e:
            print(f"[ERROR] Encoding error while reading info for {pkg}: {e}")
