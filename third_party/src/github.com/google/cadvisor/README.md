# cAdvisor

cAdvisor (Container Advisor) provides container users an understanding of the resource usage and performance characteristics of their running containers. It is a running daemon that collects, aggregates, processes, and exports information about running containers. Specifically, for each container it keeps resource isolation parameters, historical resource usage, and histograms of complete historical resource usage. This data is exported by container and machine-wide.

cAdvisor currently supports lmctfy containers as well as Docker containers (those that use the default libcontainer execdriver). Other container backends can also be added. cAdvisor's container abstraction is based on lmctfy's so containers are inherently nested hierarchically.

![cAdvisor](logo.png "cAdvisor")

#### Quick Start: Running cAdvisor in a Docker Container

To quickly tryout cAdvisor on your machine with Docker (version 0.11 or above), we have a Docker image that includes everything you need to get started. Simply run:

```
sudo docker run \
  --volume=/var/run:/var/run:rw \
  --volume=/sys/fs/cgroup/:/sys/fs/cgroup:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --publish=8080:8080 \
  --detach=true \
  google/cadvisor
```

cAdvisor is now running (in the background) on `http://localhost:8080`. The setup includes directories with Docker state cAdvisor needs to observe.

If you want to build your own cAdvisor Docker image, take a look at the Dockerfile which can build a cAdvisor docker container under `quickstart/Dockerfile`.

## Web UI

cAdvisor exposes a web UI at its port:

`http://<hostname>:<port>/`

## Remote REST API

cAdvisor exposes its raw and processed stats via a versioned remote REST API:

`http://<hostname>:<port>/api/<version>/<request>`

The current (and only) version of the API is `v1.0`.

### Version 1.0

This version exposes two main endpoints, one for container information and the other for machine information. Both endpoints are read-only in v1.0.

#### Container Information

The resource name for container information is as follows:

`/api/v1.0/containers/<absolute container name>`

Where the absolute container name follows the lmctfy naming convention. For example:

| Container Name       | Resource Name                             |
|----------------------|-------------------------------------------|
| /                    | /api/v1.0/containers/                     |
| /foo                 | /api/v1.0/containers/foo                  |
| /docker/2c4dee605d22 | /api/v1.0/containers/docker/2c4dee605d22  |

Note that the root container (`/`) contains usage for the entire machine. All Docker containers are listed under `/docker`.

The container information is returned as a JSON object containing:

- Absolute container name
- List of subcontainers
- ContainerSpec which describes the resource isolation enabled in the container
- Detailed resource usage statistics of the container for the last `N` seconds (`N` is globally configurable in cAdvisor)
- Histogram of resource usage from the creation of the container

The actual object is the marshalled JSON of the `ContainerInfo` struct found in [info/container.go](info/container.go)

#### Machine Information

The resource name for machine information is as follows:

`/api/v1.0/machine`

This resource is read-only. The machine information is returned as a JSON object containing:

- Number of schedulable logical CPU cores
- Memory capacity (in bytes)

The actual object is the marshalled JSON of the `MachineInfo` struct found in [info/machine.go](info/machine.go)

## Roadmap

cAdvisor aims to improve the resource usage and performance characteristics of running containers. Today, we gather and expose this information to users. In our roadmap:
- Advise on the performance of a container (e.g.: when it is being negatively affected by another, when it is not receiving the resources it requires, etc)
- Auto-tune the performance of the container based on previous advise.
- Provide usage prediction to cluster schedulers and orchestration layers.

## Community

Contributions, questions, and comments are all welcomed and encouraged! cAdvisor developers hand out in [#google-containers](http://webchat.freenode.net/?channels=google-containers) room on freenode.net.  We also have the [google-containers Google Groups mailing list](https://groups.google.com/forum/#!forum/google-containers).
