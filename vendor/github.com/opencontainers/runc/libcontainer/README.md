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
        "CAP_CHOWN",
        "CAP_DAC_OVERRIDE",
        "CAP_FSETID",
        "CAP_FOWNER",
        "CAP_MKNOD",
        "CAP_NET_RAW",
        "CAP_SETGID",
        "CAP_SETUID",
        "CAP_SETFCAP",
        "CAP_SETPCAP",
        "CAP_NET_BIND_SERVICE",
        "CAP_SYS_CHROOT",
        "CAP_KILL",
        "CAP_AUDIT_WRITE",
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


#### Checkpoint & Restore

libcontainer now integrates [CRIU](http://criu.org/) for checkpointing and restoring containers.
This let's you save the state of a process running inside a container to disk, and then restore
that state into a new process, on the same machine or on another machine.

`criu` version 1.5.2 or higher is required to use checkpoint and restore.
If you don't already  have `criu` installed, you can build it from source, following the
[online instructions](http://criu.org/Installation). `criu` is also installed in the docker image
generated when building libcontainer with docker.


## Copyright and license

Code and documentation copyright 2014 Docker, inc. Code released under the Apache 2.0 license.
Docs released under Creative commons.

