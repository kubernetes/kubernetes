## libcontainer - reference implementation for containers [![Build Status](https://jenkins.dockerproject.com/buildStatus/icon?job=Libcontainer Master)](https://jenkins.dockerproject.com/job/Libcontainer%20Master/) 

Libcontainer provides a native Go implementation for creating containers 
with namespaces, cgroups, capabilities, and filesystem access controls.
It allows you to manage the lifecycle of the container performing additional operations
after the container is created.


#### Container
A container is a self contained execution environment that shares the kernel of the 
host system and which is (optionally) isolated from other containers in the system.

#### Using libcontainer

To create a container you first have to initialize an instance of a factory
that will handle the creation and initialization for a container.

Because containers are spawned in a two step process you will need to provide
arguments to a binary that will be executed as the init process for the container.
To use the current binary that is spawning the containers and acting as the parent
you can use `os.Args[0]` and we have a command called `init` setup.

```go
root, err := libcontainer.New("/var/lib/container", libcontainer.InitArgs(os.Args[0], "init"))
if err != nil {
    log.Fatal(err)
}
```

Once you have an instance of the factory created we can create a configuration 
struct describing how the container is to be created.  A sample would look similar to this:

```go
config := &configs.Config{
    Rootfs: rootfs,
    Capabilities: []string{
        "CHOWN",
        "DAC_OVERRIDE",
        "FSETID",
        "FOWNER",
        "MKNOD",
        "NET_RAW",
        "SETGID",
        "SETUID",
        "SETFCAP",
        "SETPCAP",
        "NET_BIND_SERVICE",
        "SYS_CHROOT",
        "KILL",
        "AUDIT_WRITE",
    },
    Namespaces: configs.Namespaces([]configs.Namespace{
        {Type: configs.NEWNS},
        {Type: configs.NEWUTS},
        {Type: configs.NEWIPC},
        {Type: configs.NEWPID},
        {Type: configs.NEWNET},
    }),
    Cgroups: &configs.Cgroup{
        Name:            "test-container",
        Parent:          "system",
        AllowAllDevices: false,
        AllowedDevices:  configs.DefaultAllowedDevices,
    },

    Devices:  configs.DefaultAutoCreatedDevices,
    Hostname: "testing",
    Networks: []*configs.Network{
        {
            Type:    "loopback",
            Address: "127.0.0.1/0",
            Gateway: "localhost",
        },
    },
    Rlimits: []configs.Rlimit{
        {
            Type: syscall.RLIMIT_NOFILE,
            Hard: uint64(1024),
            Soft: uint64(1024),
        },
    },
}
```

Once you have the configuration populated you can create a container:

```go
container, err := root.Create("container-id", config)
```

To spawn bash as the initial process inside the container and have the
processes pid returned in order to wait, signal, or kill the process:

```go
process := &libcontainer.Process{
    Args:   []string{"/bin/bash"},
    Env:    []string{"PATH=/bin"},
    User:   "daemon",
    Stdin:  os.Stdin,
    Stdout: os.Stdout,
    Stderr: os.Stderr,
}

err := container.Start(process)
if err != nil {
    log.Fatal(err)
}

// wait for the process to finish.
status, err := process.Wait()
if err != nil {
    log.Fatal(err)
}

// destroy the container.
container.Destroy()
```

Additional ways to interact with a running container are:

```go
// return all the pids for all processes running inside the container.
processes, err := container.Processes() 

// get detailed cpu, memory, io, and network statistics for the container and 
// it's processes.
stats, err := container.Stats()


// pause all processes inside the container.
container.Pause()

// resume all paused processes.
container.Resume()
```


#### nsinit

`nsinit` is a cli application which demonstrates the use of libcontainer.  
It is able to spawn new containers or join existing containers.  A root
filesystem must be provided for use along with a container configuration file.

To build `nsinit`, run `make binary`. It will save the binary into
`bundles/nsinit`.

To use `nsinit`, cd into a Linux rootfs and copy a `container.json` file into 
the directory with your specified configuration. Environment, networking, 
and different capabilities for the container are specified in this file. 
The configuration is used for each process executed inside the container.
                                                                                                                               
See the `sample_configs` folder for examples of what the container configuration should look like.

To execute `/bin/bash` in the current directory as a container just run the following **as root**:
```bash
nsinit exec --tty /bin/bash
```

If you wish to spawn another process inside the container while your 
current bash session is running, run the same command again to 
get another bash shell (or change the command).  If the original 
process (PID 1) dies, all other processes spawned inside the container 
will be killed and the namespace will be removed. 

You can identify if a process is running in a container by 
looking to see if `state.json` is in the root of the directory.
   
You may also specify an alternate root place where 
the `container.json` file is read and where the `state.json` file will be saved.

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
