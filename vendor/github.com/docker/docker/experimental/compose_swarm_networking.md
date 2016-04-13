# Experimental: Compose, Swarm and Multi-Host Networking

The [experimental build of Docker](https://github.com/docker/docker/tree/master/experimental) has an entirely new networking system, which enables secure communication between containers on multiple hosts. In combination with Docker Swarm and Docker Compose, you can now run multi-container apps on multi-host clusters with the same tooling and configuration format you use to develop them locally.

> Note: This functionality is in the experimental stage, and contains some hacks and workarounds which will be removed as it matures.

## Prerequisites

Before you start, you’ll need to install the experimental build of Docker, and the latest versions of Machine and Compose.

-   To install the experimental Docker build on a Linux machine, follow the instructions [here](https://github.com/docker/docker/tree/master/experimental#install-docker-experimental).

-   To install the experimental Docker build on a Mac, run these commands:

        $ curl -L https://experimental.docker.com/builds/Darwin/x86_64/docker-latest > /usr/local/bin/docker
        $ chmod +x /usr/local/bin/docker

-   To install Machine, follow the instructions [here](http://docs.docker.com/machine/).

-   To install Compose, follow the instructions [here](http://docs.docker.com/compose/install/).

You’ll also need a [Docker Hub](https://hub.docker.com/account/signup/) account and a [Digital Ocean](https://www.digitalocean.com/) account.

## Set up a swarm with multi-host networking

Set the `DIGITALOCEAN_ACCESS_TOKEN` environment variable to a valid Digital Ocean API token, which you can generate in the [API panel](https://cloud.digitalocean.com/settings/applications).

    export DIGITALOCEAN_ACCESS_TOKEN=abc12345

Start a consul server:

    docker-machine --debug create \
        -d digitalocean \
        --engine-install-url="https://experimental.docker.com" \
        consul

    docker $(docker-machine config consul) run -d \
        -p "8500:8500" \
        -h "consul" \
        progrium/consul -server -bootstrap

(In a real world setting you’d set up a distributed consul, but that’s beyond the scope of this guide!)

Create a Swarm token:

    export SWARM_TOKEN=$(docker run swarm create)

Next, you create a Swarm master with Machine: 

    docker-machine --debug create \
        -d digitalocean \
        --digitalocean-image="ubuntu-14-10-x64" \
        --engine-install-url="https://experimental.docker.com" \
        --engine-opt="default-network=overlay:multihost" \
        --engine-opt="kv-store=consul:$(docker-machine ip consul):8500" \
        --engine-label="com.docker.network.driver.overlay.bind_interface=eth0" \
        swarm-0

Usually Machine can create Swarms for you, but it doesn't yet fully support multi-host networks yet, so you'll have to start up the Swarm manually:

    docker $(docker-machine config swarm-0) run -d \
        --restart="always" \
        --net="bridge" \
        swarm:latest join \
            --addr "$(docker-machine ip swarm-0):2376" \
            "token://$SWARM_TOKEN"

    docker $(docker-machine config swarm-0) run -d \
        --restart="always" \
        --net="bridge" \
        -p "3376:3376" \
        -v "/etc/docker:/etc/docker" \
        swarm:latest manage \
            --tlsverify \
            --tlscacert="/etc/docker/ca.pem" \
            --tlscert="/etc/docker/server.pem" \
            --tlskey="/etc/docker/server-key.pem" \
            -H "tcp://0.0.0.0:3376" \
            --strategy spread \
            "token://$SWARM_TOKEN"

Create a Swarm node:

    docker-machine --debug create \
        -d digitalocean \
        --digitalocean-image="ubuntu-14-10-x64" \
        --engine-install-url="https://experimental.docker.com" \
        --engine-opt="default-network=overlay:multihost" \
        --engine-opt="kv-store=consul:$(docker-machine ip consul):8500" \
        --engine-label="com.docker.network.driver.overlay.bind_interface=eth0" \
        --engine-label="com.docker.network.driver.overlay.neighbor_ip=$(docker-machine ip swarm-0)" \
        swarm-1

    docker $(docker-machine config swarm-1) run -d \
        --restart="always" \
        --net="bridge" \
        swarm:latest join \
            --addr "$(docker-machine ip swarm-1):2376" \
            "token://$SWARM_TOKEN"

You can create more Swarm nodes if you want - it’s best to give them sensible names (swarm-2, swarm-3, etc).

Finally, point Docker at your swarm:

    export DOCKER_HOST=tcp://"$(docker-machine ip swarm-0):3376"
    export DOCKER_TLS_VERIFY=1
    export DOCKER_CERT_PATH="$HOME/.docker/machine/machines/swarm-0"

## Run containers and get them communicating

Now that you’ve got a swarm up and running, you can create containers on it just like a single Docker instance:

    $ docker run busybox echo hello world
    hello world

If you run `docker ps -a`, you can see what node that container was started on by looking at its name (here it’s swarm-3):

    $ docker ps -a
    CONTAINER ID        IMAGE                      COMMAND                CREATED              STATUS                      PORTS                                   NAMES
    41f59749737b        busybox                    "echo hello world"     15 seconds ago       Exited (0) 13 seconds ago                                           swarm-3/trusting_leakey

As you start more containers, they’ll be placed on different nodes across the cluster, thanks to Swarm’s default “spread” scheduling strategy.

Every container started on this swarm will use the “overlay:multihost” network by default, meaning they can all intercommunicate. Each container gets an IP address on that network, and an `/etc/hosts` file which will be updated on-the-fly with every other container’s IP address and name. That means that if you have a running container named ‘foo’, other containers can access it at the hostname ‘foo’.

Let’s verify that multi-host networking is functioning. Start a long-running container:

    $ docker run -d --name long-running busybox top
    <container id>

If you start a new container and inspect its /etc/hosts file, you’ll see the long-running container in there:

    $ docker run busybox cat /etc/hosts
    ...
    172.21.0.6  long-running

Verify that connectivity works between containers:

    $ docker run busybox ping long-running
    PING long-running (172.21.0.6): 56 data bytes
    64 bytes from 172.21.0.6: seq=0 ttl=64 time=7.975 ms
    64 bytes from 172.21.0.6: seq=1 ttl=64 time=1.378 ms
    64 bytes from 172.21.0.6: seq=2 ttl=64 time=1.348 ms
    ^C
    --- long-running ping statistics ---
    3 packets transmitted, 3 packets received, 0% packet loss
    round-trip min/avg/max = 1.140/2.099/7.975 ms

## Run a Compose application

Here’s an example of a simple Python + Redis app using multi-host networking on a swarm.

Create a directory for the app:

    $ mkdir composetest
    $ cd composetest

Inside this directory, create 2 files.

First, create `app.py` - a simple web app that uses the Flask framework and increments a value in Redis:

    from flask import Flask
    from redis import Redis
    import os
    app = Flask(__name__)
    redis = Redis(host='composetest_redis_1', port=6379)

    @app.route('/')
    def hello():
        redis.incr('hits')
        return 'Hello World! I have been seen %s times.' % redis.get('hits')

    if __name__ == "__main__":
        app.run(host="0.0.0.0", debug=True)

Note that we’re connecting to a host called `composetest_redis_1` - this is the name of the Redis container that Compose will start.

Second, create a Dockerfile for the app container:

    FROM python:2.7
    RUN pip install flask redis
    ADD . /code
    WORKDIR /code
    CMD ["python", "app.py"]

Build the Docker image and push it to the Hub (you’ll need a Hub account). Replace `<username>` with your Docker Hub username:

    $ docker build -t <username>/counter .
    $ docker push <username>/counter

Next, create a `docker-compose.yml`, which defines the configuration for the web and redis containers. Once again, replace `<username>` with your Hub username:

    web:
      image: <username>/counter
      ports:
       - "80:5000"
    redis:
      image: redis

Now start the app:

    $ docker-compose up -d
    Pulling web (username/counter:latest)...
    swarm-0: Pulling username/counter:latest... : downloaded
    swarm-2: Pulling username/counter:latest... : downloaded
    swarm-1: Pulling username/counter:latest... : downloaded
    swarm-3: Pulling username/counter:latest... : downloaded
    swarm-4: Pulling username/counter:latest... : downloaded
    Creating composetest_web_1...
    Pulling redis (redis:latest)...
    swarm-2: Pulling redis:latest... : downloaded
    swarm-1: Pulling redis:latest... : downloaded
    swarm-3: Pulling redis:latest... : downloaded
    swarm-4: Pulling redis:latest... : downloaded
    swarm-0: Pulling redis:latest... : downloaded
    Creating composetest_redis_1...

Swarm has created containers for both web and redis, and placed them on different nodes, which you can check with `docker ps`:

    $ docker ps
    CONTAINER ID        IMAGE                      COMMAND                CREATED             STATUS              PORTS                                  NAMES
    92faad2135c9        redis                      "/entrypoint.sh redi   43 seconds ago      Up 42 seconds                                              swarm-2/composetest_redis_1
    adb809e5cdac        username/counter           "/bin/sh -c 'python    55 seconds ago      Up 54 seconds       45.67.8.9:80->5000/tcp                 swarm-1/composetest_web_1

You can also see that the web container has exposed port 80 on its swarm node. If you curl that IP, you’ll get a response from the container:

    $ curl http://45.67.8.9
    Hello World! I have been seen 1 times.

If you hit it repeatedly, the counter will increment, demonstrating that the web and redis container are communicating:

    $ curl http://45.67.8.9
    Hello World! I have been seen 2 times.
    $ curl http://45.67.8.9
    Hello World! I have been seen 3 times.
    $ curl http://45.67.8.9
    Hello World! I have been seen 4 times.
