import subprocess
#import streamlit

def run_command(command):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred while executing command: {command}\n{stderr.decode('utf-8')}")
        else:
            print(f"Command executed successfully: {command}\n{stdout.decode('utf-8')}")
    except Exception as e:
        print(f"Error occurred while executing command: {command}\n{str(e)}")

if __name__ == "__main__":
    command = "streamlit run app.py"  # Replace this with your desired command
    run_command(command)
