# Hugepages Volume

## Hugepages
Huge pages make it possible to support memory pages bigger than usually.
Link to documentation [here](https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt).

## Prerequisities
-  Pre-allocated huge pages  on the host

## Examples

### Allocating hugepages on the host

In order to allocate huge pages on the host one can:

```sh
sysctl vm.nr_hugepages=<number_of_hugepages>
```

The `/proc/meminfo` file gives an information about the total number of huge pages in the pool.
Example output of `cat /proc/meminfo | grep Huge`:

```sh
AnonHugePages:   2465792 kB
ShmemHugePages:        0 kB
HugePages_Total:     100
HugePages_Free:      100
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
```

This means that we have allocated 100 huge pages of size 2M

### Running pods

Create pod which uses Hugepages volume plugin.

Example spec:

```json
apiVersion: v1
kind: Pod
metadata:
  name: test-hugepages-volume-pod
spec:
  containers:
  - image:  gcr.io/google_containers/test-webserver
    name: test-container
    volumeMounts:
    - mountPath: /hugepages
      name: test-volume
  volumes:
  - name: test-volume
    hugePages:
      pageSize: "2M"
      maxSize: "200M"
      minSize: "100M"
```

Default value for pageSize is 2M.
All values are in DecimalSi format [link](http://physics.nist.gov/cuu/Units/binary.html).

To confirm that volume plugin mounted hugetlbfs device properly run:

```sh
mount | grep -i huge
```

There should be a hugetlbfs volume mounted into pod's sandbox:

```sh
nodev on /var/lib/kubelet/pods/bd32abe7-453d-11e7-abee-fa163e1c1bbe/volumes/kubernetes.io~hugepages/hugepage type hugetlbfs (rw,relatime,size=200M,pagesize=2M,min_size=2M
```
