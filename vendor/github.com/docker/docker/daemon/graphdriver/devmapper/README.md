# devicemapper - a storage backend based on Device Mapper

## Theory of operation

The device mapper graphdriver uses the device mapper thin provisioning
module (dm-thinp) to implement CoW snapshots. The preferred model is
to have a thin pool reserved outside of Docker and passed to the
daemon via the `--storage-opt dm.thinpooldev` option. Alternatively,
the device mapper graphdriver can setup a block device to handle this
for you via the `--storage-opt dm.directlvm_device` option.

As a fallback if no thin pool is provided, loopback files will be
created.  Loopback is very slow, but can be used without any
pre-configuration of storage.  It is strongly recommended that you do
not use loopback in production.  Ensure your Docker daemon has a
`--storage-opt dm.thinpooldev` argument provided.

In loopback, a thin pool is created at `/var/lib/docker/devicemapper`
(devicemapper graph location) based on two block devices, one for
data and one for metadata. By default these block devices are created
automatically by using loopback mounts of automatically created sparse
files.

The default loopback files used are
`/var/lib/docker/devicemapper/devicemapper/data` and
`/var/lib/docker/devicemapper/devicemapper/metadata`. Additional metadata
required to map from docker entities to the corresponding devicemapper
volumes is stored in the `/var/lib/docker/devicemapper/devicemapper/json`
file (encoded as Json).

In order to support multiple devicemapper graphs on a system, the thin
pool will be named something like: `docker-0:33-19478248-pool`, where
the `0:33` part is the minor/major device nr and `19478248` is the
inode number of the `/var/lib/docker/devicemapper` directory.

On the thin pool, docker automatically creates a base thin device,
called something like `docker-0:33-19478248-base` of a fixed
size. This is automatically formatted with an empty filesystem on
creation. This device is the base of all docker images and
containers. All base images are snapshots of this device and those
images are then in turn used as snapshots for other images and
eventually containers.

## Information on `docker info`

As of docker-1.4.1, `docker info` when using the `devicemapper` storage driver
will display something like:

	$ sudo docker info
	[...]
	Storage Driver: devicemapper
	 Pool Name: docker-253:1-17538953-pool
	 Pool Blocksize: 65.54 kB
	 Base Device Size: 107.4 GB
	 Data file: /dev/loop4
	 Metadata file: /dev/loop4
	 Data Space Used: 2.536 GB
	 Data Space Total: 107.4 GB
	 Data Space Available: 104.8 GB
	 Metadata Space Used: 7.93 MB
	 Metadata Space Total: 2.147 GB
	 Metadata Space Available: 2.14 GB
	 Udev Sync Supported: true
	 Data loop file: /home/docker/devicemapper/devicemapper/data
	 Metadata loop file: /home/docker/devicemapper/devicemapper/metadata
	 Library Version: 1.02.82-git (2013-10-04)
	[...]

### status items

Each item in the indented section under `Storage Driver: devicemapper` are
status information about the driver.
 *  `Pool Name` name of the devicemapper pool for this driver.
 *  `Pool Blocksize` tells the blocksize the thin pool was initialized with. This only changes on creation.
 *  `Base Device Size` tells the maximum size of a container and image
 *  `Data file` blockdevice file used for the devicemapper data
 *  `Metadata file` blockdevice file used for the devicemapper metadata
 *  `Data Space Used` tells how much of `Data file` is currently used
 *  `Data Space Total` tells max size the `Data file`
 *  `Data Space Available` tells how much free space there is in the `Data file`. If you are using a loop device this will report the actual space available to the loop device on the underlying filesystem.
 *  `Metadata Space Used` tells how much of `Metadata file` is currently used
 *  `Metadata Space Total` tells max size the `Metadata file`
 *  `Metadata Space Available` tells how much free space there is in the `Metadata file`. If you are using a loop device this will report the actual space available to the loop device on the underlying filesystem.
 *  `Udev Sync Supported` tells whether devicemapper is able to sync with Udev. Should be `true`.
 *  `Data loop file` file attached to `Data file`, if loopback device is used
 *  `Metadata loop file` file attached to `Metadata file`, if loopback device is used
 *  `Library Version` from the libdevmapper used

## About the devicemapper options

The devicemapper backend supports some options that you can specify
when starting the docker daemon using the `--storage-opt` flags.
This uses the `dm` prefix and would be used something like `dockerd --storage-opt dm.foo=bar`.

These options are currently documented both in [the man
page](../../../man/docker.1.md) and in [the online
documentation](https://docs.docker.com/engine/reference/commandline/dockerd/#/storage-driver-options).
If you add an options, update both the `man` page and the documentation.
