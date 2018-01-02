# <a name="linuxContainerConfiguration" />Linux Container Configuration

This document describes the schema for the [Linux-specific section](config.md#platform-specific-configuration) of the [container configuration](config.md).
The Linux container specification uses various kernel features like namespaces, cgroups, capabilities, LSM, and filesystem jails to fulfill the spec.

## <a name="configLinuxDefaultFilesystems" />Default Filesystems

The Linux ABI includes both syscalls and several special file paths.
Applications expecting a Linux environment will very likely expect these file paths to be set up correctly.

The following filesystems SHOULD be made available in each container's filesystem:

| Path     | Type   |
| -------- | ------ |
| /proc    | [procfs][] |
| /sys     | [sysfs][]  |
| /dev/pts | [devpts][] |
| /dev/shm | [tmpfs][]  |

## <a name="configLinuxNamespaces" />Namespaces

A namespace wraps a global system resource in an abstraction that makes it appear to the processes within the namespace that they have their own isolated instance of the global resource.
Changes to the global resource are visible to other processes that are members of the namespace, but are invisible to other processes.
For more information, see the [namespaces(7)][namespaces.7_2] man page.

Namespaces are specified as an array of entries inside the `namespaces` root field.
The following parameters can be specified to set up namespaces:

* **`type`** *(string, REQUIRED)* - namespace type. The following namespace types are supported:
    * **`pid`** processes inside the container will only be able to see other processes inside the same container.
    * **`network`** the container will have its own network stack.
    * **`mount`** the container will have an isolated mount table.
    * **`ipc`** processes inside the container will only be able to communicate to other processes inside the same container via system level IPC.
    * **`uts`** the container will be able to have its own hostname and domain name.
    * **`user`** the container will be able to remap user and group IDs from the host to local users and groups within the container.
    * **`cgroup`** the container will have an isolated view of the cgroup hierarchy.

* **`path`** *(string, OPTIONAL)* - an absolute path to namespace file in the [runtime mount namespace](glossary.md#runtime-namespace).
    The runtime MUST place the container process in the namespace associated with that `path`.
    The runtime MUST [generate an error](runtime.md#errors) if `path` is not associated with a namespace of type `type`.

    If `path` is not specified, the runtime MUST create a new [container namespace](glossary.md#container-namespace) of type `type`.

If a namespace type is not specified in the `namespaces` array, the container MUST inherit the [runtime namespace](glossary.md#runtime-namespace) of that type.
If a `namespaces` field contains duplicated namespaces with same `type`, the runtime MUST [generate an error](runtime.md#errors).

### Example

```json
    "namespaces": [
        {
            "type": "pid",
            "path": "/proc/1234/ns/pid"
        },
        {
            "type": "network",
            "path": "/var/run/netns/neta"
        },
        {
            "type": "mount"
        },
        {
            "type": "ipc"
        },
        {
            "type": "uts"
        },
        {
            "type": "user"
        },
        {
            "type": "cgroup"
        }
    ]
```

## <a name="configLinuxUserNamespaceMappings" />User namespace mappings

**`uidMappings`** (array of objects, OPTIONAL) describes the user namespace uid mappings from the host to the container.
**`gidMappings`** (array of objects, OPTIONAL) describes the user namespace gid mappings from the host to the container.

Each entry has the following structure:

* **`hostID`** *(uint32, REQUIRED)* - is the starting uid/gid on the host to be mapped to *containerID*.
* **`containerID`** *(uint32, REQUIRED)* - is the starting uid/gid in the container.
* **`size`** *(uint32, REQUIRED)* - is the number of ids to be mapped.

The runtime SHOULD NOT modify the ownership of referenced filesystems to realize the mapping.
Note that the number of mapping entries MAY be limited by the [kernel][user-namespaces].

### Example

```json
    "uidMappings": [
        {
            "hostID": 1000,
            "containerID": 0,
            "size": 32000
        }
    ],
    "gidMappings": [
        {
            "hostID": 1000,
            "containerID": 0,
            "size": 32000
        }
    ]
```

## <a name="configLinuxDevices" />Devices

**`devices`** (array of objects, OPTIONAL) lists devices that MUST be available in the container.
The runtime MAY supply them however it likes (with [`mknod`][mknod.2], by bind mounting from the runtime mount namespace, using symlinks, etc.).

Each entry has the following structure:

* **`type`** *(string, REQUIRED)* - type of device: `c`, `b`, `u` or `p`.
    More info in [mknod(1)][mknod.1].
* **`path`** *(string, REQUIRED)* - full path to device inside container.
    If a [file][] already exists at `path` that does not match the requested device, the runtime MUST generate an error.
* **`major, minor`** *(int64, REQUIRED unless `type` is `p`)* - [major, minor numbers][devices] for the device.
* **`fileMode`** *(uint32, OPTIONAL)* - file mode for the device.
    You can also control access to devices [with cgroups](#device-whitelist).
* **`uid`** *(uint32, OPTIONAL)* - id of device owner.
* **`gid`** *(uint32, OPTIONAL)* - id of device group.

The same `type`, `major` and `minor` SHOULD NOT be used for multiple devices.

### Example

```json
    "devices": [
        {
            "path": "/dev/fuse",
            "type": "c",
            "major": 10,
            "minor": 229,
            "fileMode": 438,
            "uid": 0,
            "gid": 0
        },
        {
            "path": "/dev/sda",
            "type": "b",
            "major": 8,
            "minor": 0,
            "fileMode": 432,
            "uid": 0,
            "gid": 0
        }
    ]
```

### <a name="configLinuxDefaultDevices" />Default Devices

In addition to any devices configured with this setting, the runtime MUST also supply:

* [`/dev/null`][null.4]
* [`/dev/zero`][zero.4]
* [`/dev/full`][full.4]
* [`/dev/random`][random.4]
* [`/dev/urandom`][random.4]
* [`/dev/tty`][tty.4]
* [`/dev/console`][console.4] is set up if terminal is enabled in the config by bind mounting the pseudoterminal slave to /dev/console.
* [`/dev/ptmx`][pts.4].
  A [bind-mount or symlink of the container's `/dev/pts/ptmx`][devpts].

## <a name="configLinuxControlGroups" />Control groups

Also known as cgroups, they are used to restrict resource usage for a container and handle device access.
cgroups provide controls (through controllers) to restrict cpu, memory, IO, pids and network for the container.
For more information, see the [kernel cgroups documentation][cgroup-v1].

### <a name="configLinuxCgroupsPath" />Cgroups Path

**`cgroupsPath`** (string, OPTIONAL) path to the cgroups.
It can be used to either control the cgroups hierarchy for containers or to run a new process in an existing container.

The value of `cgroupsPath` MUST be either an absolute path or a relative path.
* In the case of an absolute path (starting with `/`), the runtime MUST take the path to be relative to the cgroups mount point.
* In the case of a relative path (not starting with `/`), the runtime MAY interpret the path relative to a runtime-determined location in the cgroups hierarchy.

If the value is specified, the runtime MUST consistently attach to the same place in the cgroups hierarchy given the same value of `cgroupsPath`.
If the value is not specified, the runtime MAY define the default cgroups path.
Runtimes MAY consider certain `cgroupsPath` values to be invalid, and MUST generate an error if this is the case.

Implementations of the Spec can choose to name cgroups in any manner.
The Spec does not include naming schema for cgroups.
The Spec does not support per-controller paths for the reasons discussed in the [cgroupv2 documentation][cgroup-v2].
The cgroups will be created if they don't exist.

You can configure a container's cgroups via the `resources` field of the Linux configuration.
Do not specify `resources` unless limits have to be updated.
For example, to run a new process in an existing container without updating limits, `resources` need not be specified.

Runtimes MAY attach the container process to additional cgroup controllers beyond those necessary to fulfill the `resources` settings.

### Example

```json
    "cgroupsPath": "/myRuntime/myContainer",
    "resources": {
        "memory": {
        "limit": 100000,
        "reservation": 200000
        },
        "devices": [
            {
                "allow": false,
                "access": "rwm"
            }
        ]
   }
```

### <a name="configLinuxDeviceWhitelist" />Device whitelist

**`devices`** (array of objects, OPTIONAL) configures the [device whitelist][cgroup-v1-devices].
The runtime MUST apply entries in the listed order.

Each entry has the following structure:

* **`allow`** *(boolean, REQUIRED)* - whether the entry is allowed or denied.
* **`type`** *(string, OPTIONAL)* - type of device: `a` (all), `c` (char), or `b` (block).
    Unset values mean "all", mapping to `a`.
* **`major, minor`** *(int64, OPTIONAL)* - [major, minor numbers][devices] for the device.
    Unset values mean "all", mapping to [`*` in the filesystem API][cgroup-v1-devices].
* **`access`** *(string, OPTIONAL)* - cgroup permissions for device.
    A composition of `r` (read), `w` (write), and `m` (mknod).

#### Example

```json
    "devices": [
        {
            "allow": false,
            "access": "rwm"
        },
        {
            "allow": true,
            "type": "c",
            "major": 10,
            "minor": 229,
            "access": "rw"
        },
        {
            "allow": true,
            "type": "b",
            "major": 8,
            "minor": 0,
            "access": "r"
        }
    ]
```

### <a name="configLinuxMemory" />Memory

**`memory`** (object, OPTIONAL) represents the cgroup subsystem `memory` and it's used to set limits on the container's memory usage.
For more information, see the kernel cgroups documentation about [memory][cgroup-v1-memory].

Values for memory specify the limit in bytes, or `-1` for unlimited memory.

* **`limit`** *(int64, OPTIONAL)* - sets limit of memory usage
* **`reservation`** *(int64, OPTIONAL)* - sets soft limit of memory usage
* **`swap`** *(int64, OPTIONAL)* - sets limit of memory+Swap usage
* **`kernel`** *(int64, OPTIONAL)* - sets hard limit for kernel memory
* **`kernelTCP`** *(int64, OPTIONAL)* - sets hard limit for kernel TCP buffer memory

The following properties do not specify memory limits, but are covered by the `memory` controller:

* **`swappiness`** *(uint64, OPTIONAL)* - sets swappiness parameter of vmscan (See sysctl's vm.swappiness)
    The values are from 0 to 100. Higher means more swappy.
* **`disableOOMKiller`** *(bool, OPTIONAL)* - enables or disables the OOM killer.
    If enabled (`false`), tasks that attempt to consume more memory than they are allowed are immediately killed by the OOM killer.
    The OOM killer is enabled by default in every cgroup using the `memory` subsystem.
    To disable it, specify a value of `true`.

#### Example

```json
    "memory": {
        "limit": 536870912,
        "reservation": 536870912,
        "swap": 536870912,
        "kernel": -1,
        "kernelTCP": -1,
        "swappiness": 0,
        "disableOOMKiller": false
    }
```

### <a name="configLinuxCPU" />CPU

**`cpu`** (object, OPTIONAL) represents the cgroup subsystems `cpu` and `cpusets`.
For more information, see the kernel cgroups documentation about [cpusets][cgroup-v1-cpusets].

The following parameters can be specified to set up the controller:

* **`shares`** *(uint64, OPTIONAL)* - specifies a relative share of CPU time available to the tasks in a cgroup
* **`quota`** *(int64, OPTIONAL)* - specifies the total amount of time in microseconds for which all tasks in a cgroup can run during one period (as defined by **`period`** below)
* **`period`** *(uint64, OPTIONAL)* - specifies a period of time in microseconds for how regularly a cgroup's access to CPU resources should be reallocated (CFS scheduler only)
* **`realtimeRuntime`** *(int64, OPTIONAL)* - specifies a period of time in microseconds for the longest continuous period in which the tasks in a cgroup have access to CPU resources
* **`realtimePeriod`** *(uint64, OPTIONAL)* - same as **`period`** but applies to realtime scheduler only
* **`cpus`** *(string, OPTIONAL)* - list of CPUs the container will run in
* **`mems`** *(string, OPTIONAL)* - list of Memory Nodes the container will run in

#### Example

```json
    "cpu": {
        "shares": 1024,
        "quota": 1000000,
        "period": 500000,
        "realtimeRuntime": 950000,
        "realtimePeriod": 1000000,
        "cpus": "2-3",
        "mems": "0-7"
    }
```

### <a name="configLinuxBlockIO" />Block IO

**`blockIO`** (object, OPTIONAL) represents the cgroup subsystem `blkio` which implements the block IO controller.
For more information, see the kernel cgroups documentation about [blkio][cgroup-v1-blkio].

The following parameters can be specified to set up the controller:

* **`weight`** *(uint16, OPTIONAL)* - specifies per-cgroup weight. This is default weight of the group on all devices until and unless overridden by per-device rules.
* **`leafWeight`** *(uint16, OPTIONAL)* - equivalents of `weight` for the purpose of deciding how much weight tasks in the given cgroup has while competing with the cgroup's child cgroups.
* **`weightDevice`** *(array of objects, OPTIONAL)* - specifies the list of devices which will be bandwidth rate limited. The following parameters can be specified per-device:
    * **`major, minor`** *(int64, REQUIRED)* - major, minor numbers for device. More info in [mknod(1)][mknod.1] man page.
    * **`weight`** *(uint16, OPTIONAL)* - bandwidth rate for the device.
    * **`leafWeight`** *(uint16, OPTIONAL)* - bandwidth rate for the device while competing with the cgroup's child cgroups, CFQ scheduler only

    You MUST specify at least one of `weight` or `leafWeight` in a given entry, and MAY specify both.

* **`throttleReadBpsDevice`**, **`throttleWriteBpsDevice`**, **`throttleReadIOPSDevice`**, **`throttleWriteIOPSDevice`** *(array of objects, OPTIONAL)* - specify the list of devices which will be IO rate limited.
    The following parameters can be specified per-device:
    * **`major, minor`** *(int64, REQUIRED)* - major, minor numbers for device. More info in [mknod(1)][mknod.1] man page.
    * **`rate`** *(uint64, REQUIRED)* - IO rate limit for the device

#### Example

```json
    "blockIO": {
        "weight": 10,
        "leafWeight": 10,
        "weightDevice": [
            {
                "major": 8,
                "minor": 0,
                "weight": 500,
                "leafWeight": 300
            },
            {
                "major": 8,
                "minor": 16,
                "weight": 500
            }
        ],
        "throttleReadBpsDevice": [
            {
                "major": 8,
                "minor": 0,
                "rate": 600
            }
        ],
        "throttleWriteIOPSDevice": [
            {
                "major": 8,
                "minor": 16,
                "rate": 300
            }
        ]
    }
```

### <a name="configLinuxHugePageLimits" />Huge page limits

**`hugepageLimits`** (array of objects, OPTIONAL) represents the `hugetlb` controller which allows to limit the
HugeTLB usage per control group and enforces the controller limit during page fault.
For more information, see the kernel cgroups documentation about [HugeTLB][cgroup-v1-hugetlb].

Each entry has the following structure:

* **`pageSize`** *(string, REQUIRED)* - hugepage size
* **`limit`** *(uint64, REQUIRED)* - limit in bytes of *hugepagesize* HugeTLB usage

#### Example

```json
    "hugepageLimits": [
        {
            "pageSize": "2MB",
            "limit": 209715200
        }
   ]
```

### <a name="configLinuxNetwork" />Network

**`network`** (object, OPTIONAL) represents the cgroup subsystems `net_cls` and `net_prio`.
For more information, see the kernel cgroups documentations about [net\_cls cgroup][cgroup-v1-net-cls] and [net\_prio cgroup][cgroup-v1-net-prio].

The following parameters can be specified to set up the controller:

* **`classID`** *(uint32, OPTIONAL)* - is the network class identifier the cgroup's network packets will be tagged with
* **`priorities`** *(array of objects, OPTIONAL)* - specifies a list of objects of the priorities assigned to traffic originating from processes in the group and egressing the system on various interfaces.
    The following parameters can be specified per-priority:
    * **`name`** *(string, REQUIRED)* - interface name in [runtime network namespace](glossary.md#runtime-namespace)
    * **`priority`** *(uint32, REQUIRED)* - priority applied to the interface

#### Example

```json
    "network": {
        "classID": 1048577,
        "priorities": [
            {
                "name": "eth0",
                "priority": 500
            },
            {
                "name": "eth1",
                "priority": 1000
            }
        ]
   }
```

### <a name="configLinuxPIDS" />PIDs

**`pids`** (object, OPTIONAL) represents the cgroup subsystem `pids`.
For more information, see the kernel cgroups documentation about [pids][cgroup-v1-pids].

The following parameters can be specified to set up the controller:

* **`limit`** *(int64, REQUIRED)* - specifies the maximum number of tasks in the cgroup

#### Example

```json
    "pids": {
        "limit": 32771
   }
```

## <a name="configLinuxIntelRdt" />IntelRdt

**`intelRdt`** (object, OPTIONAL) represents the [Intel Resource Director Technology][intel-rdt-cat-kernel-interface].
    If `intelRdt` is set, the runtime MUST write the container process ID to the `<container-id>/tasks` file in a mounted `resctrl` pseudo-filesystem, using the container ID from [`start`](runtime.md#start) and creating the `<container-id>` directory if necessary.
    If no mounted `resctrl` pseudo-filesystem is available in the [runtime mount namespace](glossary.md#runtime-namespace), the runtime MUST [generate an error](runtime.md#errors).

    If `intelRdt` is not set, the runtime MUST NOT manipulate any `resctrl` psuedo-filesystems.

The following parameters can be specified for the container:

* **`l3CacheSchema`** *(string, OPTIONAL)* - specifies the schema for L3 cache id and capacity bitmask (CBM).
    If `l3CacheSchema` is set, runtimes MUST write the value to the `schemata` file in the `<container-id>` directory discussed in `intelRdt`.

    If `l3CacheSchema` is not set, runtimes MUST NOT write to `schemata` files in any `resctrl` psuedo-filesystems.

### Example

Consider a two-socket machine with two L3 caches where the default CBM is 0xfffff and the max CBM length is 20 bits.
Tasks inside the container only have access to the "upper" 80% of L3 cache id 0 and the "lower" 50% L3 cache id 1:

```json
"linux": {
    "intelRdt": {
        "l3CacheSchema": "L3:0=ffff0;1=3ff"
    }
}
```

## <a name="configLinuxSysctl" />Sysctl

**`sysctl`** (object, OPTIONAL) allows kernel parameters to be modified at runtime for the container.
For more information, see the [sysctl(8)][sysctl.8] man page.

### Example

```json
    "sysctl": {
        "net.ipv4.ip_forward": "1",
        "net.core.somaxconn": "256"
   }
```

## <a name="configLinuxSeccomp" />Seccomp

Seccomp provides application sandboxing mechanism in the Linux kernel.
Seccomp configuration allows one to configure actions to take for matched syscalls and furthermore also allows matching on values passed as arguments to syscalls.
For more information about Seccomp, see [Seccomp][seccomp] kernel documentation.
The actions, architectures, and operators are strings that match the definitions in seccomp.h from [libseccomp][] and are translated to corresponding values.

**`seccomp`** (object, OPTIONAL)

The following parameters can be specified to set up seccomp:

* **`defaultAction`** *(string, REQUIRED)* - the default action for seccomp. Allowed values are the same as `syscalls[].action`.

* **`architectures`** *(array of strings, OPTIONAL)* - the architecture used for system calls.
    A valid list of constants as of libseccomp v2.3.2 is shown below.

    * `SCMP_ARCH_X86`
    * `SCMP_ARCH_X86_64`
    * `SCMP_ARCH_X32`
    * `SCMP_ARCH_ARM`
    * `SCMP_ARCH_AARCH64`
    * `SCMP_ARCH_MIPS`
    * `SCMP_ARCH_MIPS64`
    * `SCMP_ARCH_MIPS64N32`
    * `SCMP_ARCH_MIPSEL`
    * `SCMP_ARCH_MIPSEL64`
    * `SCMP_ARCH_MIPSEL64N32`
    * `SCMP_ARCH_PPC`
    * `SCMP_ARCH_PPC64`
    * `SCMP_ARCH_PPC64LE`
    * `SCMP_ARCH_S390`
    * `SCMP_ARCH_S390X`
    * `SCMP_ARCH_PARISC`
    * `SCMP_ARCH_PARISC64`

* **`syscalls`** *(array of objects, OPTIONAL)* - match a syscall in seccomp.

    While this property is OPTIONAL, some values of `defaultAction` are not useful without `syscalls` entries.
    For example, if `defaultAction` is `SCMP_ACT_KILL` and `syscalls` is empty or unset, the kernel will kill the container process on its first syscall.

    Each entry has the following structure:

    * **`names`** *(array of strings, REQUIRED)* - the names of the syscalls.
        `names` MUST contain at least one entry.
    * **`action`** *(string, REQUIRED)* - the action for seccomp rules.
        A valid list of constants as of libseccomp v2.3.2 is shown below.

        * `SCMP_ACT_KILL`
        * `SCMP_ACT_TRAP`
        * `SCMP_ACT_ERRNO`
        * `SCMP_ACT_TRACE`
        * `SCMP_ACT_ALLOW`

    * **`args`** *(array of objects, OPTIONAL)* - the specific syscall in seccomp.

        Each entry has the following structure:

        * **`index`** *(uint, REQUIRED)* - the index for syscall arguments in seccomp.
        * **`value`** *(uint64, REQUIRED)* - the value for syscall arguments in seccomp.
        * **`valueTwo`** *(uint64, OPTIONAL)* - the value for syscall arguments in seccomp.
        * **`op`** *(string, REQUIRED)* - the operator for syscall arguments in seccomp.
            A valid list of constants as of libseccomp v2.3.2 is shown below.

            * `SCMP_CMP_NE`
            * `SCMP_CMP_LT`
            * `SCMP_CMP_LE`
            * `SCMP_CMP_EQ`
            * `SCMP_CMP_GE`
            * `SCMP_CMP_GT`
            * `SCMP_CMP_MASKED_EQ`

### Example

```json
    "seccomp": {
        "defaultAction": "SCMP_ACT_ALLOW",
        "architectures": [
            "SCMP_ARCH_X86",
            "SCMP_ARCH_X32"
        ],
        "syscalls": [
            {
                "names": [
                    "getcwd",
                    "chmod"
                ],
                "action": "SCMP_ACT_ERRNO"
            }
        ]
    }
```

## <a name="configLinuxRootfsMountPropagation" />Rootfs Mount Propagation

**`rootfsPropagation`** (string, OPTIONAL) sets the rootfs's mount propagation.
    Its value is either slave, private, shared or unbindable.
    The [Shared Subtrees][sharedsubtree] article in the kernel documentation has more information about mount propagation.

### Example

```json
    "rootfsPropagation": "slave",
```

## <a name="configLinuxMaskedPaths" />Masked Paths

**`maskedPaths`** (array of strings, OPTIONAL) will mask over the provided paths inside the container so that they cannot be read.
    The values MUST be absolute paths in the [container namespace](glossary.md#container_namespace).

### Example

```json
    "maskedPaths": [
        "/proc/kcore"
    ]
```

## <a name="configLinuxReadonlyPaths" />Readonly Paths

**`readonlyPaths`** (array of strings, OPTIONAL) will set the provided paths as readonly inside the container.
    The values MUST be absolute paths in the [container namespace](glossary.md#container-namespace).

### Example

```json
    "readonlyPaths": [
        "/proc/sys"
    ]
```

## <a name="configLinuxMountLabel" />Mount Label

**`mountLabel`** (string, OPTIONAL) will set the Selinux context for the mounts in the container.

### Example

```json
    "mountLabel": "system_u:object_r:svirt_sandbox_file_t:s0:c715,c811"
```


[cgroup-v1]: https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt
[cgroup-v1-blkio]: https://www.kernel.org/doc/Documentation/cgroup-v1/blkio-controller.txt
[cgroup-v1-cpusets]: https://www.kernel.org/doc/Documentation/cgroup-v1/cpusets.txt
[cgroup-v1-devices]: https://www.kernel.org/doc/Documentation/cgroup-v1/devices.txt
[cgroup-v1-hugetlb]: https://www.kernel.org/doc/Documentation/cgroup-v1/hugetlb.txt
[cgroup-v1-memory]: https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt
[cgroup-v1-net-cls]: https://www.kernel.org/doc/Documentation/cgroup-v1/net_cls.txt
[cgroup-v1-net-prio]: https://www.kernel.org/doc/Documentation/cgroup-v1/net_prio.txt
[cgroup-v1-pids]: https://www.kernel.org/doc/Documentation/cgroup-v1/pids.txt
[cgroup-v2]: https://www.kernel.org/doc/Documentation/cgroup-v2.txt
[devices]: https://www.kernel.org/doc/Documentation/admin-guide/devices.txt
[devpts]: https://www.kernel.org/doc/Documentation/filesystems/devpts.txt
[file]: http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap03.html#tag_03_164
[libseccomp]: https://github.com/seccomp/libseccomp
[procfs]: https://www.kernel.org/doc/Documentation/filesystems/proc.txt
[seccomp]: https://www.kernel.org/doc/Documentation/prctl/seccomp_filter.txt
[sharedsubtree]: https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt
[sysfs]: https://www.kernel.org/doc/Documentation/filesystems/sysfs.txt
[tmpfs]: https://www.kernel.org/doc/Documentation/filesystems/tmpfs.txt

[console.4]: http://man7.org/linux/man-pages/man4/console.4.html
[full.4]: http://man7.org/linux/man-pages/man4/full.4.html
[mknod.1]: http://man7.org/linux/man-pages/man1/mknod.1.html
[mknod.2]: http://man7.org/linux/man-pages/man2/mknod.2.html
[namespaces.7_2]: http://man7.org/linux/man-pages/man7/namespaces.7.html
[null.4]: http://man7.org/linux/man-pages/man4/null.4.html
[pts.4]: http://man7.org/linux/man-pages/man4/pts.4.html
[random.4]: http://man7.org/linux/man-pages/man4/random.4.html
[sysctl.8]: http://man7.org/linux/man-pages/man8/sysctl.8.html
[tty.4]: http://man7.org/linux/man-pages/man4/tty.4.html
[zero.4]: http://man7.org/linux/man-pages/man4/zero.4.html
[user-namespaces]: http://man7.org/linux/man-pages/man7/user_namespaces.7.html
[intel-rdt-cat-kernel-interface]: https://www.kernel.org/doc/Documentation/x86/intel_rdt_ui.txt
