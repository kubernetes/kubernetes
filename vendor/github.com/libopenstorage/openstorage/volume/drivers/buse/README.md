## What is BUSE?
BUSE is a block driver in user space.  It leverages `NBD` to provide block volume access to a container.  In the back, it writes data out to a local file.

### Using BUSE
Declare `buse` as a driver in your OSD config file as such:
```
---
osd:
  cluster:
    nodeid: "1"
    clusterid: "deadbeeef"
  drivers:
    buse:
```

BUSE relies on NBD to export block devices.  Therefore, remember to `modprobe nbd`.
