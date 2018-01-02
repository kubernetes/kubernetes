# Container Lifecycle

While containerd is a daemon that provides API to manage multiple containers, the containers themselves are not tied to the lifecycle of containerd.  Each container has a shim that acts as the direct parent for the container's processes as well as reporting the exit status and holding onto the STDIO of the container.  This also allows containerd to crash and restore all functionality to containers.


## containerd

The daemon provides an API to manage multiple containers.  It can handle locking in process where needed to coordinate tasks between subsystems.  While the daemon does fork off the needed processes to run containers, the shim and runc, these are re-parented to the system's init.

## shim

Each container has its own shim that acts as the direct parent of the container's processes.  The shim is responsible for keeping the IO and/or pty master of the container open, writing the container's exit status for containerd, and reaping the container's processes when they exit.  Since the shim owns the container's pty master, it provides an API for resizing.

Overall, a container's lifecycle is not tied to the containerd daemon.  The daemon is a management API for multiple container whose lifecycle is tied to one shim per container.
