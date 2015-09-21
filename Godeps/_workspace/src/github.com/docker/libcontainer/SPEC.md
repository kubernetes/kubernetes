## Container Specification - v1

This is the standard configuration for version 1 containers.  It includes
namespaces, standard filesystem setup, a default Linux capability set, and
information about resource reservations.  It also has information about any 
populated environment settings for the processes running inside a container.

Along with the configuration of how a container is created the standard also
discusses actions that can be performed on a container to manage and inspect
information about the processes running inside.

The v1 profile is meant to be able to accommodate the majority of applications
with a strong security configuration.

### System Requirements and Compatibility

Minimum requirements:
* Kernel version - 3.8 recommended 2.6.2x minimum(with backported patches) 
* Mounted cgroups with each subsystem in its own hierarchy


### Namespaces

|     Flag      | Enabled | 
| ------------  | ------- |
| CLONE_NEWPID  |    1    |
| CLONE_NEWUTS  |    1    |
| CLONE_NEWIPC  |    1    |
| CLONE_NEWNET  |    1    |
| CLONE_NEWNS   |    1    |
| CLONE_NEWUSER |    0    |

In v1 the user namespace is not enabled by default for support of older kernels
where the user namespace feature is not fully implemented.  Namespaces are 
created for the container via the `clone` syscall.  


### Filesystem

A root filesystem must be provided to a container for execution.  The container
will use this root filesystem (rootfs) to jail and spawn processes inside where
the binaries and system libraries are local to that directory.  Any binaries
to be executed must be contained within this rootfs.

Mounts that happen inside the container are automatically cleaned up when the
container exits as the mount namespace is destroyed and the kernel will 
unmount all the mounts that were setup within that namespace.

For a container to execute properly there are certain filesystems that 
are required to be mounted within the rootfs that the runtime will setup.

|     Path    |  Type  |                  Flags                 |                 Data                    |
| ----------- | ------ | -------------------------------------- | --------------------------------------- |
| /proc       | proc   | MS_NOEXEC,MS_NOSUID,MS_NODEV           |                                         |
| /dev        | tmpfs  | MS_NOEXEC,MS_STRICTATIME               | mode=755                                |
| /dev/shm    | shm    | MS_NOEXEC,MS_NOSUID,MS_NODEV           | mode=1777,size=65536k                   |
| /dev/mqueue | mqueue | MS_NOEXEC,MS_NOSUID,MS_NODEV           |                                         |
| /dev/pts    | devpts | MS_NOEXEC,MS_NOSUID                    | newinstance,ptmxmode=0666,mode=620,gid5 |
| /sys        | sysfs  | MS_NOEXEC,MS_NOSUID,MS_NODEV,MS_RDONLY |                                         |


After a container's filesystems are mounted within the newly created 
mount namespace `/dev` will need to be populated with a set of device nodes.
It is expected that a rootfs does not need to have any device nodes specified
for `/dev` witin the rootfs as the container will setup the correct devices
that are required for executing a container's process.

|      Path    | Mode |   Access   |
| ------------ | ---- | ---------- |
| /dev/null    | 0666 |  rwm       |
| /dev/zero    | 0666 |  rwm       |
| /dev/full    | 0666 |  rwm       |
| /dev/tty     | 0666 |  rwm       |
| /dev/random  | 0666 |  rwm       |
| /dev/urandom | 0666 |  rwm       |
| /dev/fuse    | 0666 |  rwm       |


**ptmx**
`/dev/ptmx` will need to be a symlink to the host's `/dev/ptmx` within
the container.  

The use of a pseudo TTY is optional within a container and it should support both.
If a pseudo is provided to the container `/dev/console` will need to be 
setup by binding the console in `/dev/` after it has been populated and mounted
in tmpfs.

|      Source     | Destination  | UID GID | Mode | Type |
| --------------- | ------------ | ------- | ---- | ---- |
| *pty host path* | /dev/console | 0 0     | 0600 | bind | 


After `/dev/null` has been setup we check for any external links between
the container's io, STDIN, STDOUT, STDERR.  If the container's io is pointing
to `/dev/null` outside the container we close and `dup2` the the `/dev/null` 
that is local to the container's rootfs.


After the container has `/proc` mounted a few standard symlinks are setup 
within `/dev/` for the io.

|    Source    | Destination |
| ------------ | ----------- |
| /proc/1/fd   | /dev/fd     |
| /proc/1/fd/0 | /dev/stdin  |
| /proc/1/fd/1 | /dev/stdout |
| /proc/1/fd/2 | /dev/stderr |

A `pivot_root` is used to change the root for the process, effectively 
jailing the process inside the rootfs.

```c
put_old = mkdir(...);
pivot_root(rootfs, put_old);
chdir("/");
unmount(put_old, MS_DETACH);
rmdir(put_old);
```

For container's running with a rootfs inside `ramfs` a `MS_MOVE` combined
with a `chroot` is required as `pivot_root` is not supported in `ramfs`.

```c
mount(rootfs, "/", NULL, MS_MOVE, NULL);
chroot(".");
chdir("/");
```

The `umask` is set back to `0022` after the filesystem setup has been completed.

### Resources

Cgroups are used to handle resource allocation for containers.  This includes
system resources like cpu, memory, and device access.

| Subsystem  | Enabled |
| ---------- | ------- |
| devices    | 1       |
| memory     | 1       |
| cpu        | 1       |
| cpuacct    | 1       |
| cpuset     | 1       |
| blkio      | 1       |
| perf_event | 1       |
| freezer    | 1       |


All cgroup subsystem are joined so that statistics can be collected from
each of the subsystems.  Freezer does not expose any stats but is joined
so that containers can be paused and resumed.

The parent process of the container's init must place the init pid inside
the correct cgroups before the initialization begins.  This is done so
that no processes or threads escape the cgroups.  This sync is 
done via a pipe ( specified in the runtime section below ) that the container's
init process will block waiting for the parent to finish setup.

### Security 

The standard set of Linux capabilities that are set in a container
provide a good default for security and flexibility for the applications.


|     Capability       | Enabled |
| -------------------- | ------- |
| CAP_NET_RAW          | 1       |
| CAP_NET_BIND_SERVICE | 1       |
| CAP_AUDIT_WRITE      | 1       |
| CAP_DAC_OVERRIDE     | 1       |
| CAP_SETFCAP          | 1       |
| CAP_SETPCAP          | 1       |
| CAP_SETGID           | 1       |
| CAP_SETUID           | 1       |
| CAP_MKNOD            | 1       |
| CAP_CHOWN            | 1       |
| CAP_FOWNER           | 1       |
| CAP_FSETID           | 1       |
| CAP_KILL             | 1       |
| CAP_SYS_CHROOT       | 1       |
| CAP_NET_BROADCAST    | 0       |
| CAP_SYS_MODULE       | 0       |
| CAP_SYS_RAWIO        | 0       |
| CAP_SYS_PACCT        | 0       |
| CAP_SYS_ADMIN        | 0       |
| CAP_SYS_NICE         | 0       |
| CAP_SYS_RESOURCE     | 0       |
| CAP_SYS_TIME         | 0       |
| CAP_SYS_TTY_CONFIG   | 0       |
| CAP_AUDIT_CONTROL    | 0       |
| CAP_MAC_OVERRIDE     | 0       |
| CAP_MAC_ADMIN        | 0       |
| CAP_NET_ADMIN        | 0       |
| CAP_SYSLOG           | 0       |
| CAP_DAC_READ_SEARCH  | 0       |
| CAP_LINUX_IMMUTABLE  | 0       |
| CAP_IPC_LOCK         | 0       |
| CAP_IPC_OWNER        | 0       |
| CAP_SYS_PTRACE       | 0       |
| CAP_SYS_BOOT         | 0       |
| CAP_LEASE            | 0       |
| CAP_WAKE_ALARM       | 0       |
| CAP_BLOCK_SUSPE      | 0       |


Additional security layers like [apparmor](https://wiki.ubuntu.com/AppArmor)
and [selinux](http://selinuxproject.org/page/Main_Page) can be used with
the containers.  A container should support setting an apparmor profile or 
selinux process and mount labels if provided in the configuration.  

Standard apparmor profile:
```c
#include <tunables/global>
profile <profile_name> flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  network,
  capability,
  file,
  umount,

  mount fstype=tmpfs,
  mount fstype=mqueue,
  mount fstype=fuse.*,
  mount fstype=binfmt_misc -> /proc/sys/fs/binfmt_misc/,
  mount fstype=efivarfs -> /sys/firmware/efi/efivars/,
  mount fstype=fusectl -> /sys/fs/fuse/connections/,
  mount fstype=securityfs -> /sys/kernel/security/,
  mount fstype=debugfs -> /sys/kernel/debug/,
  mount fstype=proc -> /proc/,
  mount fstype=sysfs -> /sys/,

  deny @{PROC}/sys/fs/** wklx,
  deny @{PROC}/sysrq-trigger rwklx,
  deny @{PROC}/mem rwklx,
  deny @{PROC}/kmem rwklx,
  deny @{PROC}/sys/kernel/[^s][^h][^m]* wklx,
  deny @{PROC}/sys/kernel/*/** wklx,

  deny mount options=(ro, remount) -> /,
  deny mount fstype=debugfs -> /var/lib/ureadahead/debugfs/,
  deny mount fstype=devpts,

  deny /sys/[^f]*/** wklx,
  deny /sys/f[^s]*/** wklx,
  deny /sys/fs/[^c]*/** wklx,
  deny /sys/fs/c[^g]*/** wklx,
  deny /sys/fs/cg[^r]*/** wklx,
  deny /sys/firmware/efi/efivars/** rwklx,
  deny /sys/kernel/security/** rwklx,
}
```

*TODO: seccomp work is being done to find a good default config*

### Runtime and Init Process

During container creation the parent process needs to talk to the container's init 
process and have a form of synchronization.  This is accomplished by creating
a pipe that is passed to the container's init.  When the init process first spawns 
it will block on its side of the pipe until the parent closes its side.  This
allows the parent to have time to set the new process inside a cgroup hierarchy 
and/or write any uid/gid mappings required for user namespaces.  
The pipe is passed to the init process via FD 3.

The application consuming libcontainer should be compiled statically.  libcontainer
does not define any init process and the arguments provided are used to `exec` the
process inside the application.  There should be no long running init within the 
container spec.

If a pseudo tty is provided to a container it will open and `dup2` the console
as the container's STDIN, STDOUT, STDERR as well as mounting the console
as `/dev/console`.

An extra set of mounts are provided to a container and setup for use.  A container's
rootfs can contain some non portable files inside that can cause side effects during
execution of a process.  These files are usually created and populated with the container
specific information via the runtime.  

**Extra runtime files:**
* /etc/hosts 
* /etc/resolv.conf
* /etc/hostname
* /etc/localtime


#### Defaults

There are a few defaults that can be overridden by users, but in their omission
these apply to processes within a container.

|       Type          |             Value              |
| ------------------- | ------------------------------ |
| Parent Death Signal | SIGKILL                        | 
| UID                 | 0                              |
| GID                 | 0                              |
| GROUPS              | 0, NULL                        |
| CWD                 | "/"                            |
| $HOME               | Current user's home dir or "/" |
| Readonly rootfs     | false                          |
| Pseudo TTY          | false                          |


## Actions

After a container is created there is a standard set of actions that can
be done to the container.  These actions are part of the public API for 
a container.

|     Action     |                         Description                                |
| -------------- | ------------------------------------------------------------------ |
| Get processes  | Return all the pids for processes running inside a container       | 
| Get Stats      | Return resource statistics for the container as a whole            |
| Wait           | Wait waits on the container's init process ( pid 1 )               |
| Wait Process   | Wait on any of the container's processes returning the exit status | 
| Destroy        | Kill the container's init process and remove any filesystem state  |
| Signal         | Send a signal to the container's init process                      |
| Signal Process | Send a signal to any of the container's processes                  |
| Pause          | Pause all processes inside the container                           |
| Resume         | Resume all processes inside the container if paused                |
| Exec           | Execute a new process inside of the container  ( requires setns )  |

### Execute a new process inside of a running container.

User can execute a new process inside of a running container. Any binaries to be
executed must be accessible within the container's rootfs.

The started process will run inside the container's rootfs. Any changes
made by the process to the container's filesystem will persist after the
process finished executing.

The started process will join all the container's existing namespaces. When the
container is paused, the process will also be paused and will resume when
the container is unpaused.  The started process will only run when the container's
primary process (PID 1) is running, and will not be restarted when the container
is restarted.

#### Planned additions

The started process will have its own cgroups nested inside the container's
cgroups. This is used for process tracking and optionally resource allocation
handling for the new process. Freezer cgroup is required, the rest of the cgroups
are optional. The process executor must place its pid inside the correct
cgroups before starting the process. This is done so that no child processes or
threads can escape the cgroups.

When the process is stopped, the process executor will try (in a best-effort way)
to stop all its children and remove the sub-cgroups.
