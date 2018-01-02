# <a name="containerConfigurationFile" />Container Configuration file

This configuration file contains metadata necessary to implement [standard operations](runtime.md#operations) against the container.
This includes the process to run, environment variables to inject, sandboxing features to use, etc.

The canonical schema is defined in this document, but there is a JSON Schema in [`schema/config-schema.json`](schema/config-schema.json) and Go bindings in [`specs-go/config.go`](specs-go/config.go).
[Platform](spec.md#platforms)-specific configuration schema are defined in the [platform-specific documents](#platform-specific-configuration) linked below.
For properties that are only defined for some [platforms](spec.md#platforms), the Go property has a `platform` tag listing those protocols (e.g. `platform:"linux,solaris"`).

Below is a detailed description of each field defined in the configuration format and valid values are specified.
Platform-specific fields are identified as such.
For all platform-specific configuration values, the scope defined below in the [Platform-specific configuration](#platform-specific-configuration) section applies.


## <a name="configSpecificationVersion" />Specification version

* **`ociVersion`** (string, REQUIRED) MUST be in [SemVer v2.0.0][semver-v2.0.0] format and specifies the version of the Open Container Runtime Specification with which the bundle complies.
    The Open Container Runtime Specification follows semantic versioning and retains forward and backward compatibility within major versions.
    For example, if a configuration is compliant with version 1.1 of this specification, it is compatible with all runtimes that support any 1.1 or later release of this specification, but is not compatible with a runtime that supports 1.0 and not 1.1.

### Example

```json
    "ociVersion": "0.1.0"
```

## <a name="configRoot" />Root

**`root`** (object, OPTIONAL) specifies the container's root filesystem.
    On Windows, for Windows Server Containers, this field is REQUIRED.
    For [Hyper-V Containers](config-windows.md#hyperv), this field MUST NOT be set.

    On all other platforms, this field is REQUIRED.

* **`path`** (string, REQUIRED) Specifies the path to the root filesystem for the container.

    * On Windows, `path` MUST be a [volume GUID path][naming-a-volume].

    * On POSIX platforms, `path` is either an absolute path or a relative path to the bundle.
        For example, with a bundle at `/to/bundle` and a root filesystem at `/to/bundle/rootfs`, the `path` value can be either `/to/bundle/rootfs` or `rootfs`.
        The value SHOULD be the conventional `rootfs`.

    A directory MUST exist at the path declared by the field.

* **`readonly`** (bool, OPTIONAL) If true then the root filesystem MUST be read-only inside the container, defaults to false.
    * On Windows, this field MUST be omitted or false.

### Example (POSIX platforms)

```json
"root": {
    "path": "rootfs",
    "readonly": true
}
```

### Example (Windows)

```json
"root": {
    "path": "\\\\?\\Volume{ec84d99e-3f02-11e7-ac6c-00155d7682cf}\\"
}
```

## <a name="configMounts" />Mounts

**`mounts`** (array of objects, OPTIONAL) specifies additional mounts beyond [`root`](#root).
    The runtime MUST mount entries in the listed order.
    For Linux, the parameters are as documented in [mount(2)][mount.2] system call man page.
    For Solaris, the mount entry corresponds to the 'fs' resource in the [zonecfg(1M)][zonecfg.1m] man page.

* **`destination`** (string, REQUIRED) Destination of mount point: path inside container.
    This value MUST be an absolute path.
    * Windows: one mount destination MUST NOT be nested within another mount (e.g., c:\\foo and c:\\foo\\bar).
    * Solaris: corresponds to "dir" of the fs resource in [zonecfg(1M)][zonecfg.1m].
* **`source`** (string, OPTIONAL) A device name, but can also be a directory name or a dummy.
    Path values are either absolute or relative to the bundle.
    * Windows: a local directory on the filesystem of the container host. UNC paths and mapped drives are not supported.
    * Solaris: corresponds to "special" of the fs resource in [zonecfg(1M)][zonecfg.1m].
* **`options`** (array of strings, OPTIONAL) Mount options of the filesystem to be used.
    * Linux: supported options are listed in the [mount(8)][mount.8] man page.
      Note both [filesystem-independent][mount.8-filesystem-independent] and [filesystem-specific][mount.8-filesystem-specific] options are listed.
    * Solaris: corresponds to "options" of the fs resource in [zonecfg(1M)][zonecfg.1m].
    * Windows: runtimes MUST support `ro`, mounting the filesystem read-only when `ro` is given.

### Example (Windows)

```json
"mounts": [
    {
        "destination": "C:\\folder-inside-container",
        "source": "C:\\folder-on-host",
        "options": ["ro"]
    }
]
```

### <a name="configPOSIXMounts" />POSIX-platform Mounts

For POSIX platforms the `mounts` structure has the following fields:

* **`type`** (string, OPTIONAL) The type of the filesystem to be mounted.
  * Linux: filesystem types supported by the kernel as listed in */proc/filesystems* (e.g., "minix", "ext2", "ext3", "jfs", "xfs", "reiserfs", "msdos", "proc", "nfs", "iso9660").
  * Solaris: corresponds to "type" of the fs resource in [zonecfg(1M)][zonecfg.1m].

### Example (Linux)

```json
"mounts": [
    {
        "destination": "/tmp",
        "type": "tmpfs",
        "source": "tmpfs",
        "options": ["nosuid","strictatime","mode=755","size=65536k"]
    },
    {
        "destination": "/data",
        "type": "bind",
        "source": "/volumes/testing",
        "options": ["rbind","rw"]
    }
]
```

### Example (Solaris)

```json
"mounts": [
    {
        "destination": "/opt/local",
        "type": "lofs",
        "source": "/usr/local",
        "options": ["ro","nodevices"]
    },
    {
        "destination": "/opt/sfw",
        "type": "lofs",
        "source": "/opt/sfw"
    }
]
```

## <a name="configProcess" />Process

**`process`** (object, OPTIONAL) specifies the container process.
    This property is REQUIRED when [`start`](runtime.md#start) is called.

* **`terminal`** (bool, OPTIONAL) specifies whether a terminal is attached to the process, defaults to false.
    As an example, if set to true on Linux a pseudoterminal pair is allocated for the process and the pseudoterminal slave is duplicated on the process's [standard streams][stdin.3].
* **`consoleSize`** (object, OPTIONAL) specifies the console size in characters of the terminal.
    Runtimes MUST ignore `consoleSize` if `terminal` is `false` or unset.
    * **`height`** (uint, REQUIRED)
    * **`width`** (uint, REQUIRED)
* **`cwd`** (string, REQUIRED) is the working directory that will be set for the executable.
    This value MUST be an absolute path.
* **`env`** (array of strings, OPTIONAL) with the same semantics as [IEEE Std 1003.1-2008's `environ`][ieee-1003.1-2008-xbd-c8.1].
* **`args`** (array of strings, REQUIRED) with similar semantics to [IEEE Std 1003.1-2008 `execvp`'s *argv*][ieee-1003.1-2008-xsh-exec].
    This specification extends the IEEE standard in that at least one entry is REQUIRED, and that entry is used with the same semantics as `execvp`'s *file*.

### <a name="configPOSIXProcess" />POSIX process

For systems that support POSIX rlimits (for example Linux and Solaris), the `process` object supports the following process-specific properties:

* **`rlimits`** (array of objects, OPTIONAL) allows setting resource limits for the process.
    Each entry has the following structure:

    * **`type`** (string, REQUIRED) the platform resource being limited.
        * Linux: valid values are defined in the [`getrlimit(2)`][getrlimit.2] man page, such as `RLIMIT_MSGQUEUE`.
        * Solaris: valid values are defined in the [`getrlimit(3)`][getrlimit.3] man page, such as `RLIMIT_CORE`.

        The runtime MUST [generate an error](runtime.md#errors) for any values which cannot be mapped to a relevant kernel interface
        For each entry in `rlimits`, a [`getrlimit(3)`][getrlimit.3] on `type` MUST succeed.
        For the following properties, `rlim` refers to the status returned by the `getrlimit(3)` call.

    * **`soft`** (uint64, REQUIRED) the value of the limit enforced for the corresponding resource.
        `rlim.rlim_cur` MUST match the configured value.
    * **`hard`** (uint64, REQUIRED) the ceiling for the soft limit that could be set by an unprivileged process.
        `rlim.rlim_max` MUST match the configured value.
        Only a privileged process (e.g. one with the `CAP_SYS_RESOURCE` capability) can raise a hard limit.

    If `rlimits` contains duplicated entries with same `type`, the runtime MUST [generate an error](runtime.md#errors).

### <a name="configLinuxProcess" />Linux Process

For Linux-based systems, the `process` object supports the following process-specific properties.

* **`apparmorProfile`** (string, OPTIONAL) specifies the name of the AppArmor profile for the process.
    For more information about AppArmor, see [AppArmor documentation][apparmor].
* **`capabilities`** (object, OPTIONAL) is an object containing arrays that specifies the sets of capabilities for the process.
    Valid values are defined in the [capabilities(7)][capabilities.7] man page, such as `CAP_CHOWN`.
    Any value which cannot be mapped to a relevant kernel interface MUST cause an error.
    `capabilities` contains the following properties:

    * **`effective`** (array of strings, OPTIONAL) the `effective` field is an array of effective capabilities that are kept for the process.
    * **`bounding`** (array of strings, OPTIONAL) the `bounding` field is an array of bounding capabilities that are kept for the process.
    * **`inheritable`** (array of strings, OPTIONAL) the `inheritable` field is an array of inheritable capabilities that are kept for the process.
    * **`permitted`** (array of strings, OPTIONAL) the `permitted` field is an array of permitted capabilities that are kept for the process.
    * **`ambient`** (array of strings, OPTIONAL) the `ambient` field is an array of ambient capabilities that are kept for the process.
* **`noNewPrivileges`** (bool, OPTIONAL) setting `noNewPrivileges` to true prevents the process from gaining additional privileges.
    As an example, the [`no_new_privs`][no-new-privs] article in the kernel documentation has information on how this is achieved using a `prctl` system call on Linux.
* **`oomScoreAdj`** *(int, OPTIONAL)* adjusts the oom-killer score in `[pid]/oom_score_adj` for the process's `[pid]` in a [proc pseudo-filesystem][procfs].
    If `oomScoreAdj` is set, the runtime MUST set `oom_score_adj` to the given value.
    If `oomScoreAdj` is not set, the runtime MUST NOT change the value of `oom_score_adj`.

    This is a per-process setting, where as [`disableOOMKiller`](config-linux.md#memory) is scoped for a memory cgroup.
    For more information on how these two settings work together, see [the memory cgroup documentation section 10. OOM Contol][cgroup-v1-memory_2].
* **`selinuxLabel`** (string, OPTIONAL) specifies the SELinux label for the process.
    For more information about SELinux, see  [SELinux documentation][selinux].

### <a name="configUser" />User

The user for the process is a platform-specific structure that allows specific control over which user the process runs as.

#### <a name="configPOSIXUser" />POSIX-platform User

For POSIX platforms the `user` structure has the following fields:

* **`uid`** (int, REQUIRED) specifies the user ID in the [container namespace](glossary.md#container-namespace).
* **`gid`** (int, REQUIRED) specifies the group ID in the [container namespace](glossary.md#container-namespace).
* **`additionalGids`** (array of ints, OPTIONAL) specifies additional group IDs in the [container namespace](glossary.md#container-namespace) to be added to the process.

_Note: symbolic name for uid and gid, such as uname and gname respectively, are left to upper levels to derive (i.e. `/etc/passwd` parsing, NSS, etc)_

### Example (Linux)

```json
"process": {
    "terminal": true,
    "consoleSize": {
        "height": 25,
        "width": 80
    },
    "user": {
        "uid": 1,
        "gid": 1,
        "additionalGids": [5, 6]
    },
    "env": [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "TERM=xterm"
    ],
    "cwd": "/root",
    "args": [
        "sh"
    ],
    "apparmorProfile": "acme_secure_profile",
    "selinuxLabel": "system_u:system_r:svirt_lxc_net_t:s0:c124,c675",
    "noNewPrivileges": true,
    "capabilities": {
        "bounding": [
            "CAP_AUDIT_WRITE",
            "CAP_KILL",
            "CAP_NET_BIND_SERVICE"
        ],
       "permitted": [
            "CAP_AUDIT_WRITE",
            "CAP_KILL",
            "CAP_NET_BIND_SERVICE"
        ],
       "inheritable": [
            "CAP_AUDIT_WRITE",
            "CAP_KILL",
            "CAP_NET_BIND_SERVICE"
        ],
        "effective": [
            "CAP_AUDIT_WRITE",
            "CAP_KILL"
        ],
        "ambient": [
            "CAP_NET_BIND_SERVICE"
        ]
    },
    "rlimits": [
        {
            "type": "RLIMIT_NOFILE",
            "hard": 1024,
            "soft": 1024
        }
    ]
}
```
### Example (Solaris)

```json
"process": {
    "terminal": true,
    "consoleSize": {
        "height": 25,
        "width": 80
    },
    "user": {
        "uid": 1,
        "gid": 1,
        "additionalGids": [2, 8]
    },
    "env": [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "TERM=xterm"
    ],
    "cwd": "/root",
    "args": [
        "/usr/bin/bash"
    ]
}
```

#### <a name="configWindowsUser" />Windows User

For Windows based systems the user structure has the following fields:

* **`username`** (string, OPTIONAL) specifies the user name for the process.

### Example (Windows)

```json
"process": {
    "terminal": true,
    "user": {
        "username": "containeradministrator"
    },
    "env": [
        "VARIABLE=1"
    ],
    "cwd": "c:\\foo",
    "args": [
        "someapp.exe",
    ]
}
```


## <a name="configHostname" />Hostname

* **`hostname`** (string, OPTIONAL) specifies the container's hostname as seen by processes running inside the container.
    On Linux, for example, this will change the hostname in the [container](glossary.md#container-namespace) [UTS namespace][uts-namespace.7].
    Depending on your [namespace configuration](config-linux.md#namespaces), the container UTS namespace may be the [runtime](glossary.md#runtime-namespace) [UTS namespace][uts-namespace.7].

### Example

```json
"hostname": "mrsdalloway"
```

## <a name="configPlatformSpecificConfiguration" />Platform-specific configuration

* **`linux`** (object, OPTIONAL) [Linux-specific configuration](config-linux.md).
    This MAY be set if the target platform of this spec is `linux`.
* **`windows`** (object, OPTIONAL) [Windows-specific configuration](config-windows.md).
    This MUST be set if the target platform of this spec is `windows`.
* **`solaris`** (object, OPTIONAL) [Solaris-specific configuration](config-solaris.md).
    This MAY be set if the target platform of this spec is `solaris`.

### Example (Linux)

```json
{
    "linux": {
        "namespaces": [
            {
                "type": "pid"
            }
        ]
    }
}
```

## <a name="configHooks" />POSIX-platform Hooks

For POSIX platforms, the configuration structure supports `hooks` for configuring custom actions related to the [lifecycle](runtime.md#lifecycle) of the container.

* **`hooks`** (object, OPTIONAL) MAY contain any of the following properties:
    * **`prestart`** (array of objects, OPTIONAL) is an array of [pre-start hooks](#prestart).
        Entries in the array contain the following properties:
        * **`path`** (string, REQUIRED) with similar semantics to [IEEE Std 1003.1-2008 `execv`'s *path*][ieee-1003.1-2008-functions-exec].
            This specification extends the IEEE standard in that **`path`** MUST be absolute.
        * **`args`** (array of strings, OPTIONAL) with the same semantics as [IEEE Std 1003.1-2008 `execv`'s *argv*][ieee-1003.1-2008-functions-exec].
        * **`env`** (array of strings, OPTIONAL) with the same semantics as [IEEE Std 1003.1-2008's `environ`][ieee-1003.1-2008-xbd-c8.1].
        * **`timeout`** (int, OPTIONAL) is the number of seconds before aborting the hook.
            If set, `timeout` MUST be greater than zero.
    * **`poststart`** (array of objects, OPTIONAL) is an array of [post-start hooks](#poststart).
        Entries in the array have the same schema as pre-start entries.
    * **`poststop`** (array of objects, OPTIONAL) is an array of [post-stop hooks](#poststop).
        Entries in the array have the same schema as pre-start entries.

Hooks allow users to specify programs to run before or after various lifecycle events.
Hooks MUST be called in the listed order.
The [state](runtime.md#state) of the container MUST be passed to hooks over stdin so that they may do work appropriate to the current state of the container.

### <a name="configHooksPrestart" />Prestart

The pre-start hooks MUST be called after the [`start`](runtime.md#start) operation is called but [before the user-specified program command is executed](runtime.md#lifecycle).
On Linux, for example, they are called after the container namespaces are created, so they provide an opportunity to customize the container (e.g. the network namespace could be specified in this hook).

### <a name="configHooksPoststart" />Poststart

The post-start hooks MUST be called [after the user-specified process is executed](runtime.md#lifecycle) but before the [`start`](runtime.md#start) operation returns.
For example, this hook can notify the user that the container process is spawned.

### <a name="configHooksPoststop" />Poststop

The post-stop hooks MUST be called [after the container is deleted](runtime.md#lifecycle) but before the [`delete`](runtime.md#delete) operation returns.
Cleanup or debugging functions are examples of such a hook.

### Example

```json
    "hooks": {
        "prestart": [
            {
                "path": "/usr/bin/fix-mounts",
                "args": ["fix-mounts", "arg1", "arg2"],
                "env":  [ "key1=value1"]
            },
            {
                "path": "/usr/bin/setup-network"
            }
        ],
        "poststart": [
            {
                "path": "/usr/bin/notify-start",
                "timeout": 5
            }
        ],
        "poststop": [
            {
                "path": "/usr/sbin/cleanup.sh",
                "args": ["cleanup.sh", "-f"]
            }
        ]
    }
```

## <a name="configAnnotations" />Annotations

**`annotations`** (object, OPTIONAL) contains arbitrary metadata for the container.
    This information MAY be structured or unstructured.
    Annotations MUST be a key-value map.
    If there are no annotations then this property MAY either be absent or an empty map.

    Keys MUST be strings.
    Keys MUST NOT be an empty string.
    Keys SHOULD be named using a reverse domain notation - e.g. `com.example.myKey`.
    Keys using the `org.opencontainers` namespace are reserved and MUST NOT be used by subsequent specifications.
    Implementations that are reading/processing this configuration file MUST NOT generate an error if they encounter an unknown annotation key.

    Values MUST be strings.
    Values MAY be an empty string.

```json
"annotations": {
    "com.example.gpu-cores": "2"
}
```

## <a name="configExtensibility" />Extensibility

Runtimes that are reading or processing this configuration file MUST NOT generate an error if they encounter an unknown property.
Instead they MUST ignore unknown properties.

## Valid values

Runtimes that are reading or processing this configuration file MUST generate an error when invalid or unsupported values are encountered.
Unless support for a valid value is explicitly required, runtimes MAY choose which subset of the valid values it will support.

## Configuration Schema Example

Here is a full example `config.json` for reference.

```json
{
    "ociVersion": "0.5.0-dev",
    "process": {
        "terminal": true,
        "user": {
            "uid": 1,
            "gid": 1,
            "additionalGids": [
                5,
                6
            ]
        },
        "args": [
            "sh"
        ],
        "env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "TERM=xterm"
        ],
        "cwd": "/",
        "capabilities": {
            "bounding": [
                "CAP_AUDIT_WRITE",
                "CAP_KILL",
                "CAP_NET_BIND_SERVICE"
            ],
            "permitted": [
                "CAP_AUDIT_WRITE",
                "CAP_KILL",
                "CAP_NET_BIND_SERVICE"
            ],
            "inheritable": [
                "CAP_AUDIT_WRITE",
                "CAP_KILL",
                "CAP_NET_BIND_SERVICE"
            ],
            "effective": [
                "CAP_AUDIT_WRITE",
                "CAP_KILL"
            ],
            "ambient": [
                "CAP_NET_BIND_SERVICE"
            ]
        },
        "rlimits": [
            {
                "type": "RLIMIT_CORE",
                "hard": 1024,
                "soft": 1024
            },
            {
                "type": "RLIMIT_NOFILE",
                "hard": 1024,
                "soft": 1024
            }
        ],
        "apparmorProfile": "acme_secure_profile",
        "oomScoreAdj": 100,
        "selinuxLabel": "system_u:system_r:svirt_lxc_net_t:s0:c124,c675",
        "noNewPrivileges": true
    },
    "root": {
        "path": "rootfs",
        "readonly": true
    },
    "hostname": "slartibartfast",
    "mounts": [
        {
            "destination": "/proc",
            "type": "proc",
            "source": "proc"
        },
        {
            "destination": "/dev",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": [
                "nosuid",
                "strictatime",
                "mode=755",
                "size=65536k"
            ]
        },
        {
            "destination": "/dev/pts",
            "type": "devpts",
            "source": "devpts",
            "options": [
                "nosuid",
                "noexec",
                "newinstance",
                "ptmxmode=0666",
                "mode=0620",
                "gid=5"
            ]
        },
        {
            "destination": "/dev/shm",
            "type": "tmpfs",
            "source": "shm",
            "options": [
                "nosuid",
                "noexec",
                "nodev",
                "mode=1777",
                "size=65536k"
            ]
        },
        {
            "destination": "/dev/mqueue",
            "type": "mqueue",
            "source": "mqueue",
            "options": [
                "nosuid",
                "noexec",
                "nodev"
            ]
        },
        {
            "destination": "/sys",
            "type": "sysfs",
            "source": "sysfs",
            "options": [
                "nosuid",
                "noexec",
                "nodev"
            ]
        },
        {
            "destination": "/sys/fs/cgroup",
            "type": "cgroup",
            "source": "cgroup",
            "options": [
                "nosuid",
                "noexec",
                "nodev",
                "relatime",
                "ro"
            ]
        }
    ],
    "hooks": {
        "prestart": [
            {
                "path": "/usr/bin/fix-mounts",
                "args": [
                    "fix-mounts",
                    "arg1",
                    "arg2"
                ],
                "env": [
                    "key1=value1"
                ]
            },
            {
                "path": "/usr/bin/setup-network"
            }
        ],
        "poststart": [
            {
                "path": "/usr/bin/notify-start",
                "timeout": 5
            }
        ],
        "poststop": [
            {
                "path": "/usr/sbin/cleanup.sh",
                "args": [
                    "cleanup.sh",
                    "-f"
                ]
            }
        ]
    },
    "linux": {
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
        ],
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
        ],
        "sysctl": {
            "net.ipv4.ip_forward": "1",
            "net.core.somaxconn": "256"
        },
        "cgroupsPath": "/myRuntime/myContainer",
        "resources": {
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
            },
            "pids": {
                "limit": 32771
            },
            "hugepageLimits": [
                {
                    "pageSize": "2MB",
                    "limit": 9223372036854772000
                }
            ],
            "memory": {
                "limit": 536870912,
                "reservation": 536870912,
                "swap": 536870912,
                "kernel": -1,
                "kernelTCP": -1,
                "swappiness": 0,
                "disableOOMKiller": false
            },
            "cpu": {
                "shares": 1024,
                "quota": 1000000,
                "period": 500000,
                "realtimeRuntime": 950000,
                "realtimePeriod": 1000000,
                "cpus": "2-3",
                "mems": "0-7"
            },
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
            ],
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
        },
        "rootfsPropagation": "slave",
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
        },
        "namespaces": [
            {
                "type": "pid"
            },
            {
                "type": "network"
            },
            {
                "type": "ipc"
            },
            {
                "type": "uts"
            },
            {
                "type": "mount"
            },
            {
                "type": "user"
            },
            {
                "type": "cgroup"
            }
        ],
        "maskedPaths": [
            "/proc/kcore",
            "/proc/latency_stats",
            "/proc/timer_stats",
            "/proc/sched_debug"
        ],
        "readonlyPaths": [
            "/proc/asound",
            "/proc/bus",
            "/proc/fs",
            "/proc/irq",
            "/proc/sys",
            "/proc/sysrq-trigger"
        ],
        "mountLabel": "system_u:object_r:svirt_sandbox_file_t:s0:c715,c811"
    },
    "annotations": {
        "com.example.key1": "value1",
        "com.example.key2": "value2"
    }
}
```


[apparmor]: https://wiki.ubuntu.com/AppArmor
[cgroup-v1-memory_2]: https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt
[selinux]:http://selinuxproject.org/page/Main_Page
[no-new-privs]: https://www.kernel.org/doc/Documentation/prctl/no_new_privs.txt
[procfs_2]: https://www.kernel.org/doc/Documentation/filesystems/proc.txt
[semver-v2.0.0]: http://semver.org/spec/v2.0.0.html
[ieee-1003.1-2008-xbd-c8.1]: http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap08.html#tag_08_01
[ieee-1003.1-2008-xsh-exec]: http://pubs.opengroup.org/onlinepubs/9699919799/functions/exec.html
[naming-a-volume]: https://aka.ms/nb3hqb

[capabilities.7]: http://man7.org/linux/man-pages/man7/capabilities.7.html
[mount.2]: http://man7.org/linux/man-pages/man2/mount.2.html
[mount.8]: http://man7.org/linux/man-pages/man8/mount.8.html
[mount.8-filesystem-independent]: http://man7.org/linux/man-pages/man8/mount.8.html#FILESYSTEM-INDEPENDENT_MOUNT%20OPTIONS
[mount.8-filesystem-specific]: http://man7.org/linux/man-pages/man8/mount.8.html#FILESYSTEM-SPECIFIC_MOUNT%20OPTIONS
[getrlimit.2]: http://man7.org/linux/man-pages/man2/getrlimit.2.html
[getrlimit.3]: http://pubs.opengroup.org/onlinepubs/9699919799/functions/getrlimit.html
[stdin.3]: http://man7.org/linux/man-pages/man3/stdin.3.html
[uts-namespace.7]: http://man7.org/linux/man-pages/man7/namespaces.7.html
[zonecfg.1m]: http://docs.oracle.com/cd/E86824_01/html/E54764/zonecfg-1m.html
