## ChainFS Graph Driver

### EXPERIMENTAL!  
Note that this is still in development and experimental.

### About
The `chainfs` graph driver leverages `libchainfs` (https://github.com/libopenstorage/chainfs) user space layered filesystem to create a storage of graph layers.

`libchainfs` can we configured to use an object store interface to read, write and create snapshots.  The provider of this interface is passed in via the `chainfs.volume_driver` parameter when starting Docker.

It also uses an optimized way of computing the layer diffs and avoids using the NaiveDiff implementation.

To use this as the graphdriver in Docker with btrfs as the backend volume provider:

```
DOCKER_STORAGE_OPTIONS= -s chainfs --storage-opt chainfs.volume_driver=btrfs
```

or

```
docker daemon --storage-driver=chainfs --storage-opt chainfs.volume_driver=btrfs
```

Note that the `chainfs.volume_driver` parameter will indicate the actual volume provider to be used by chainfs.  `chainfs` exposes a set of interfaces that will be called by `libchainfs` to perform the `create`, `read`, `write`, `truncate` object store operations.

### Building

#### Installing Dependancies
1. Make sure you have `fuse` installed.
2. Make sure you have `libchainfs` installed.  Visit https://github.com/libopenstorage/chainfs to obtain `libchainfs`.

#### To build OSD
When building `OSD`, run:

```
HAVE_CHAINFS=1 EXPERIMENTAL_=1 make
```

### Running
Here is a sample `config.yaml` file to load the `chainfs` graph driver:

```
---
osd:
  cluster:
     nodeid: "1"
     clusterid: "deadbeeef"
  drivers:
     btrfs:
  graphdrivers:
     chainfs:
```

#### Note
Set your ulimit -n to run out of open files.  Graph drivers tend to have many concurrently open files.
