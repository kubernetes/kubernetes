# Running cAdvisor

## With Docker

We have a Docker image that includes everything you need to get started. Simply run:

```
sudo docker run \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:rw \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --publish=8080:8080 \
  --detach=true \
  --name=cadvisor \
  google/cadvisor:latest
```

cAdvisor is now running (in the background) on `http://localhost:8080/`. The setup includes directories with Docker state cAdvisor needs to observe.

## Latest Canary

The latest cAdvisor canary release is continuously built from HEAD and available
as an Automated Build Docker image:
[google/cadvisor-canary](https://registry.hub.docker.com/u/google/cadvisor-canary/). We do *not* recommend using this image in production due to its large size and volatility.

## With Boot2Docker

After booting up a boot2docker instance, run cAdvisor image with the same docker command mentioned above. cAdvisor can now be accessed at port 8080 of your boot2docker instance. The host IP can be found through DOCKER_HOST environment variable setup by boot2docker:

```
$ echo $DOCKER_HOST
tcp://192.168.59.103:2376
```

In this case, cAdvisor UI should be accessible on `http://192.168.59.103:8080`

## Other Configurations

### CentOS, Fedora, and RHEL

You may need to run the container with `--privileged=true` and `--volume=/cgroup:/cgroup:ro \` in order for cAdvisor to monitor Docker containers.

RHEL and CentOS lock down their containers a bit more. cAdvisor needs access to the Docker daemon through its socket. This requires `--privileged=true` in RHEL and CentOS.

On some versions of RHEL and CentOS the cgroup hierarchies are mounted in `/cgroup` so run cAdvisor with an additional Docker option of `--volume=/cgroup:/cgroup:ro \`.

### Debian

By default, Debian disables the memory cgroup which does not allow cAdvisor to gather memory stats. To enable the memory cgroup take a look at [these instructions](https://github.com/google/cadvisor/issues/432).

### LXC Docker exec driver

If you are using Docker with the LXC exec driver, then you need to manually specify all cgroup mounts by adding the:

```
  --volume=/cgroup/cpu:/cgroup/cpu \
  --volume=/cgroup/cpuacct:/cgroup/cpuacct \
  --volume=/cgroup/cpuset:/cgroup/cpuset \
  --volume=/cgroup/memory:/cgroup/memory \
  --volume=/cgroup/blkio:/cgroup/blkio \
```

### Invalid Bindmount `/`

This is a problem seen in older versions of Docker. To fix, start cAdvisor without the `--volume=/:/rootfs:ro` mount. cAdvisor will degrade gracefully by dropping stats that depend on access to the machine root.

## Standalone

cAdvisor is a static Go binary with no external dependencies. To run it standalone all you should need to do is run it! Note that some data sources may require root priviledges. cAdvisor will gracefully degrade its features to those it can expose with the access given.

```
$ sudo cadvisor
```

cAdvisor is now running (in the foreground) on `http://localhost:8080/`.

## Runtime Options

cAdvisor has a series of flags that can be used to configure its runtime behavior. More details can be found in runtime [options](runtime_options.md).

## I need help!

We aim to have cAdvisor run everywhere! If you run into issues getting it running, feel free to file an issue. We are very responsive in supporting our users and update our documentation with new setups.
