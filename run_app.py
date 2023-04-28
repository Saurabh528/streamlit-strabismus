import os
import sys
import streamlit.cli as stcli

if __name__ == "__main__":
        sys.argv = ["streamlit", "run", "app.py"]
        sys.exit(stcli.main())

