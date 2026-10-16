"""PyInstaller entrypoint. The application opens a native window, no server."""
from deadline_engine.desktop import main

if __name__ == "__main__":
    main()
