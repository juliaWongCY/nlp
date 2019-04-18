# Docker Deployment Instructions

## For Windows using Docker for Windows (CE)
* Prerequisites:
    * Routing table **must** be updated to include the virtual vm switch
      that docker uses (DockerNAT).
    * You can find the network switch address by first finding the
      correct network hosted by docker:
        ```
        docker network ls
        ```
    * Which by default should have a `bridge` network
    * Find the (subnet) address by running an inspection:
        ```
        docker network inspect bridge
        ```
        (By default, it should be `172.17.0.0`)
    * You should also be able to find the address of your container
      (e.g. `172.17.0.2`)
    * Add the subnet address to your host routing table manually by:
        ```
        route add 172.17.0.0 mask 255.255.0.0 <address_of_container> -p
        ```
      replacing `<address_of_container>` with the actual address


To be able to receive response from the container, you will need to
expose the container's port either via an `EXPOSE <port>` instruction
in the Dockerfile or `--expose` command flag when starting the container.
