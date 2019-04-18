# Deployment script for Windows hosts

Write-Output "[Deploy] Start"

if ($(docker ps -q -f name=nlp_backend)) {
    docker stop nlp_backend
}

if ($(docker ps -aq -f name=nlp_backend)) {
    docker rm nlp_backend
}

Write-Output "Building nlp_backend container"

docker build -f Dockerfile.win -t nlp_backend .

if ($?) {
    Write-Output "Starting Docker Container: nlp_backend"
    docker run -d -p 8000:6543 --name nlp_backend nlp_backend
} else {
    Write-Error "Docker container build was unsuccessful!"
}

Write-Output "[Deploy] End"
