# Dockerfile for GlusterFS
Forked from https://github.com/gluster/docker

# Running

```bash
$ sudo docker run -d \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v /run/lvm:/run/lvm \
  -v /dev:/dev \
  -v /var/lib/heketi:/var/lib/heketi \
  --privileged  \
  heketi/gluster
```

# Build

```
# docker build --rm --tag heketi/gluster:dev . 
```


