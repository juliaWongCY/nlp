# VS Code Build Task
# Requirements and build process:
# 1. Sets python environment to local virtual @ (proj_root)/venv
# 2. Install dependencies indicated by (proj_root)/tmp2/tmp2.egg-info/requires.txt
# 3. Runs test by executing 'py.test.exe'

# To be executed in project root folder

Write-Output '[Test] runtests.ps1'

# Set virtual environment
./venv/Scripts/activate
if ($?) {
    Write-Output 'pip found at: ' $(pip --version)
} else {
    Write-Output 'pip (Virtual Environment) not found'
    Exit 1
}

Write-Output 'Installing test dependencies...'
Set-Location tmp2
pip install -e '.[testing]'

if ($?) {
    Write-Output 'Dependencies Installed'
} else {
    Write-Output 'Error in installing dependencies'
    Exit 1
}

# Run Tests
py.test.exe

if ($?) {
    Write-Output '[Test] All tests passed!'
    Set-Location ../
    ./venv/Scripts/deactivate
} else {
    Write-Output '[Test] Tests failures :('
    Set-Location ../
    ./venv/Scripts/deactivate
    Exit 1
}
