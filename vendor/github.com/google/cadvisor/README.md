# cAdvisor

cAdvisor (Container Advisor) provides container users an understanding of the resource usage and performance characteristics of their running containers. It is a running daemon that collects, aggregates, processes, and exports information about running containers. Specifically, for each container it keeps resource isolation parameters, historical resource usage, histograms of complete historical resource usage and network statistics. This data is exported by container and machine-wide.

cAdvisor has native support for [Docker](https://github.com/docker/docker) containers and should support just about any other container type out of the box. We strive for support across the board so feel free to open an issue if that is not the case. cAdvisor's container abstraction is based on [lmctfy](https://github.com/google/lmctfy)'s so containers are inherently nested hierarchically.

![cAdvisor](logo.png "cAdvisor")

#### Quick Start: Running cAdvisor in a Docker Container

To quickly tryout cAdvisor on your machine with Docker, we have a Docker image that includes everything you need to get started. You can run a single cAdvisor to monitor the whole machine. Simply run:

```
sudo docker run \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:rw \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --volume=/dev/disk/:/dev/disk:ro \
  --publish=8080:8080 \
  --detach=true \
  --name=cadvisor \
  google/cadvisor:latest
```

cAdvisor is now running (in the background) on `http://localhost:8080`. The setup includes directories with Docker state cAdvisor needs to observe.

**Note**: If you're running on CentOS, Fedora, RHEL, or are using LXC take a look at our [running instructions](docs/running.md).

We have detailed [instructions](docs/running.md#standalone) on running cAdvisor standalone outside of Docker. cAdvisor [running options](docs/runtime_options.md) may also be interesting for advanced usecases. If you want to build your own cAdvisor Docker image see our [deployment](docs/deploy.md) page.

## Building and Testing

See the more detailed instructions in the [build page](docs/development/build.md). This includes instructions for building and deploying the cAdvisor Docker image.

## Exporting stats

cAdvisor supports exporting stats to various storage plugins. See the [documentation](docs/storage/README.md) for more details and examples.

## Web UI

cAdvisor exposes a web UI at its port:

`http://<hostname>:<port>/`

See the [documentation](docs/web.md) for more details.

## Remote REST API & Clients

cAdvisor exposes its raw and processed stats via a versioned remote REST API. See the API's [documentation](docs/api.md) for more information.

There is also an official Go client implementation in the [client](client/) directory. See the [documentation](docs/clients.md) for more information.

## Roadmap

cAdvisor aims to improve the resource usage and performance characteristics of running containers. Today, we gather and expose this information to users. In our roadmap:
- Advise on the performance of a container (e.g.: when it is being negatively affected by another, when it is not receiving the resources it requires, etc)
- Auto-tune the performance of the container based on previous advise.
- Provide usage prediction to cluster schedulers and orchestration layers.

## Community

Contributions, questions, and comments are all welcomed and encouraged! cAdvisor developers hang out on [Slack](https://kubernetes.slack.com) in the #sig-node channel (get an invitation [here](http://slack.kubernetes.io/)). We also have the [kubernetes-users Google Groups mailing list](https://groups.google.com/forum/#!forum/kubernetes-users).
