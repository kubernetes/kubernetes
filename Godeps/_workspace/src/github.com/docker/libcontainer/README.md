## libcontainer - reference implementation for containers [![Build Status](https://ci.dockerproject.com/github.com/docker/libcontainer/status.svg?branch=master)](https://ci.dockerproject.com/github.com/docker/libcontainer) 

### Note on API changes:

Please bear with us while we work on making the libcontainer API stable and something that we can support long term.  We are currently discussing the API with the community, therefore, if you currently depend on libcontainer please pin your dependency at a specific tag or commit id.  Please join the discussion and help shape the API.

#### Background

libcontainer specifies configuration options for what a container is.  It provides a native Go implementation for using Linux namespaces with no external dependencies.  libcontainer provides many convenience functions for working with namespaces, networking, and management.  


#### Container
A container is a self contained execution environment that shares the kernel of the host system and which is (optionally) isolated from other containers in the system.

libcontainer may be used to execute a process in a container. If a user tries to run a new process inside an existing container, the new process is added to the processes executing in the container.


#### Root file system

A container runs with a directory known as its *root file system*, or *rootfs*, mounted as the file system root. The rootfs is usually a full system tree.


#### Configuration

A container is initially configured by supplying configuration data when the container is created.


#### nsinit

`nsinit` is a cli application which demonstrates the use of libcontainer.  It is able to spawn new containers or join existing containers, based on the current directory.

To use `nsinit`, cd into a Linux rootfs and copy a `container.json` file into the directory with your specified configuration. Environment, networking, and different capabilities for the container are specified in this file. The configuration is used for each process executed inside the container.
                                                                                                                               
See the `sample_configs` folder for examples of what the container configuration should look like.

To execute `/bin/bash` in the current directory as a container just run the following **as root**:
```bash
nsinit exec /bin/bash
```

If you wish to spawn another process inside the container while your current bash session is running, run the same command again to get another bash shell (or change the command).  If the original process (PID 1) dies, all other processes spawned inside the container will be killed and the namespace will be removed. 

You can identify if a process is running in a container by looking to see if `state.json` is in the root of the directory.
   
You may also specify an alternate root place where the `container.json` file is read and where the `state.json` file will be saved.

#### Future
See the [roadmap](ROADMAP.md).

## Copyright and license

Code and documentation copyright 2014 Docker, inc. Code released under the Apache 2.0 license.
Docs released under Creative commons.

## Hacking on libcontainer

First of all, please familiarise yourself with the [libcontainer Principles](PRINCIPLES.md).

If you're a *contributor* or aspiring contributor, you should read the [Contributors' Guide](CONTRIBUTING.md).

If you're a *maintainer* or aspiring maintainer, you should read the [Maintainers' Guide](MAINTAINERS_GUIDE.md) and
"How can I become a maintainer?" in the Contributors' Guide.
