<!--[metadata]>
+++
title = "Volume plugins"
description = "How to manage data with external volume plugins"
keywords = ["Examples, Usage, volume, docker, data, volumes, plugin, api"]
[menu.main]
parent = "mn_extend"
+++
<![end-metadata]-->

# Write a volume plugin

Docker volume plugins enable Docker deployments to be integrated with external
storage systems, such as Amazon EBS, and enable data volumes to persist beyond
the lifetime of a single Docker host. See the [plugin documentation](plugins.md)
for more information.

# Command-line changes

A volume plugin makes use of the `-v`and `--volume-driver` flag on the `docker run` command.  The `-v` flag accepts a volume name and the `--volume-driver` flag a driver type, for example: 

    $ docker run -ti -v volumename:/data --volume-driver=flocker   busybox sh

This command passes the `volumename` through to the volume plugin as a
user-given name for the volume. The `volumename` must not begin with a `/`. 

By having the user specify a  `volumename`, a plugin can associate the volume
with an external volume beyond the lifetime of a single container or container
host. This can be used, for example, to move a stateful container from one
server to another.

By specifying a `volumedriver` in conjunction with a `volumename`, users can use plugins such as [Flocker](https://clusterhq.com/docker-plugin/) to manage volumes external to a single host, such as those on EBS. 


# Create a VolumeDriver

The container creation endpoint (`/containers/create`) accepts a `VolumeDriver`
field of type `string` allowing to specify the name of the driver. It's default
value of `"local"` (the default driver for local volumes).

# Volume plugin protocol

If a plugin registers itself as a `VolumeDriver` when activated, then it is
expected to provide writeable paths on the host filesystem for the Docker
daemon to provide to containers to consume.

The Docker daemon handles bind-mounting the provided paths into user
containers.

### /VolumeDriver.Create

**Request**:
```
{
    "Name": "volume_name"
}
```

Instruct the plugin that the user wants to create a volume, given a user
specified volume name.  The plugin does not need to actually manifest the
volume on the filesystem yet (until Mount is called).

**Response**:
```
{
    "Err": null
}
```

Respond with a string error if an error occurred.

### /VolumeDriver.Remove

**Request**:
```
{
    "Name": "volume_name"
}
```

Create a volume, given a user specified volume name.

**Response**:
```
{
    "Err": null
}
```

Respond with a string error if an error occurred.

### /VolumeDriver.Mount

**Request**:
```
{
    "Name": "volume_name"
}
```

Docker requires the plugin to provide a volume, given a user specified volume
name. This is called once per container start.

**Response**:
```
{
    "Mountpoint": "/path/to/directory/on/host",
    "Err": null
}
```

Respond with the path on the host filesystem where the volume has been made
available, and/or a string error if an error occurred.

### /VolumeDriver.Path

**Request**:
```
{
    "Name": "volume_name"
}
```

Docker needs reminding of the path to the volume on the host.

**Response**:
```
{
    "Mountpoint": "/path/to/directory/on/host",
    "Err": null
}
```

Respond with the path on the host filesystem where the volume has been made
available, and/or a string error if an error occurred.

### /VolumeDriver.Unmount

**Request**:
```
{
    "Name": "volume_name"
}
```

Indication that Docker no longer is using the named volume. This is called once
per container stop.  Plugin may deduce that it is safe to deprovision it at
this point.

**Response**:
```
{
    "Err": null
}
```

Respond with a string error if an error occurred.

