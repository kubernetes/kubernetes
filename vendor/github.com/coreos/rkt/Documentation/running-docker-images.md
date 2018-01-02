# Running Docker images with rkt

rkt features native support for fetching and running Docker container images.

## Getting started

To reference a Docker image, use the `docker://` prefix when fetching or running images.

Note that Docker images do not support signature verification, and hence it's necessary to use the `--insecure-options=image` flag.

As a simple example, let's run the latest [`redis`][docker-redis] container image from the default Docker registry:

```
# rkt --insecure-options=image run docker://redis
rkt: fetching image from docker://redis
rkt: warning: image signature verification has been disabled
Downloading layer: 511136ea3c5a64f264b78b5433614aec563103b4d4702f3ba7d4d2698e22c158
...
Downloading layer: f2fb89b0a711a7178528c7785d247ba3572924353b0d5e23e9b28f0518253b22
4:C 19 Apr 06:09:02.372 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
4:M 19 Apr 06:09:02.373 # You requested maxclients of 10000 requiring at least 10032 max file descriptors.
4:M 19 Apr 06:09:02.373 # Redis can't set maximum open files to 10032 because of OS error: Operation not permitted.
4:M 19 Apr 06:09:02.373 # Current maximum open files is 8192. maxclients has been reduced to 8160 to compensate for low ulimit. If you need higher maxclients increase 'ulimit -n'.
                _._
           _.-``__ ''-._
      _.-``    `.  `_.  ''-._           Redis 3.0.0 (00000000/0) 64 bit
  .-`` .-```.  ```\/    _.,_ ''-._
 (    '      ,       .-`  | `,    )     Running in standalone mode
 |`-._`-...-` __...-.``-._|'` _.-'|     Port: 6379
 |    `-._   `._    /     _.-'    |     PID: 4
  `-._    `-._  `-./  _.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |           http://redis.io
  `-._    `-._`-.__.-'_.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |
  `-._    `-._`-.__.-'_.-'    _.-'
      `-._    `-.__.-'    _.-'
          `-._        _.-'
              `-.__.-'

4:M 19 Apr 06:09:02.374 # Server started, Redis version 3.0.0
4:M 19 Apr 06:09:02.375 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
4:M 19 Apr 06:09:02.375 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
4:M 19 Apr 06:09:02.375 * The server is now ready to accept connections on port 6379
```

This behaves similarly to the Docker client: if no specific registry is named, the [Docker Hub][docker-hub] is used by default.

As with Docker, alternative registries can be used by specifying the registry as part of the image reference.
For example, the following command will fetch an [nginx][quay-nginx] Docker image hosted on [quay.io][quay]:

```
# rkt --insecure-options=image fetch docker://quay.io/zanui/nginx
rkt: fetching image from docker://quay.io/zanui/nginx
rkt: warning: image signature verification has been disabled
Downloading layer: 511136ea3c5a64f264b78b5433614aec563103b4d4702f3ba7d4d2698e22c158
...
Downloading layer: 340951f1240f3dc1189ae32cfa5af35df2dc640e0c92f2397b7a72e174c1a158
sha512-c6d6efd98f506380ff128e473ca239ed
```

The hash printed in the final line represents the image ID of the converted ACI.

After the image has been retrieved, it can be run by referencing this hash:

```
# rkt --insecure-options=image run sha512-c6d6efd98f506380ff128e473ca239ed
```


[docker-redis]: https://hub.docker.com/_/redis/
[docker-hub]: https://hub.docker.com
[quay]: https://quay.io/
[quay-nginx]: https://quay.io/repository/zanui/nginx
