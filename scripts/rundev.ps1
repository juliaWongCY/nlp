# VS Code Build Task
# Requirements and build process:
# 1. Sets python environment to local virtual @ (proj_root)/venv
# 2. Install dependencies indicated by (proj_root)/tmp2/tmp2.egg-info/requires.txt
# 3. Starts server by running 'pserve development.ini'

# To be executed in project root folder

Write-Output '[Build] rundev.ps1'

# Set virtual environment
./venv/Scripts/activate
if ($?) {
    Write-Output 'pip found at: ' $(pip --version)
} else {
    Write-Output 'pip (Virtual Environment) not found'
    Exit 1
}

Write-Output 'Installing dependencies...'
Set-Location tmp2
pip install -e .

if ($?) {
    Write-Output 'Dependencies Installed'
} else {
    Write-Output 'Error in installing dependencies'
    Exit 1
}

../venv/Scripts/pserve.exe development.ini
