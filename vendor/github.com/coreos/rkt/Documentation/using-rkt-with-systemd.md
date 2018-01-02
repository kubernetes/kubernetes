# Using rkt with systemd

`rkt` is designed to cooperate with init systems, like [`systemd`][systemd]. rkt implements a simple CLI that directly executes processes, and does not interpose a long-running daemon, so the lifecycle of rkt pods can be directly managed by systemd. Standard systemd idioms like `systemctl start` and `systemctl stop` work out of the box.

In the shell excerpts below, a `#` prompt indicates commands that require root privileges, while the `$` prompt denotes commands issued as an unprivileged user.

## systemd-run

The [`systemd-run`][systemd-run] utility is a convenient shortcut for testing a service before making it permanent in a unit file. To start a "daemonized" container that forks the container processes into the background, wrap the invocation of `rkt` with `systemd-run`:

```
# systemd-run --slice=machine rkt run coreos.com/etcd:v2.2.5
Running as unit run-29486.service.
```

The `--slice=machine` option to `systemd-run` places the service in `machine.slice` rather than the host's `system.slice`, isolating containers in their own cgroup area.

Invoking a rkt container through systemd-run in this way creates a transient service unit that can be managed with the usual systemd tools:

```
$ systemctl status run-29486.service
● run-29486.service - /bin/rkt run coreos.com/etcd:v2.2.5
   Loaded: loaded (/run/systemd/system/run-29486.service; static; vendor preset: disabled)
  Drop-In: /run/systemd/system/run-29486.service.d
           └─50-Description.conf, 50-ExecStart.conf, 50-Slice.conf
   Active: active (running) since Wed 2016-02-24 12:50:20 CET; 27s ago
 Main PID: 29487 (ld-linux-x86-64)
   Memory: 36.1M
      CPU: 1.467s
   CGroup: /machine.slice/run-29486.service
           ├─29487 stage1/rootfs/usr/lib/ld-linux-x86-64.so.2 stage1/rootfs/usr/bin/systemd-nspawn --boot -Zsystem_u:system_r:svirt_lxc_net_t:s0:c46...
           ├─29535 /usr/lib/systemd/systemd --default-standard-output=tty --log-target=null --log-level=warning --show-status=0
           └─system.slice
             ├─etcd.service
             │ └─29544 /etcd
             └─systemd-journald.service
               └─29539 /usr/lib/systemd/systemd-journald
```

Since every pod is registered with [`machined`][machined] with a machine name of the form `rkt-$UUID`, the systemd tools can inspect pod logs, or stop and restart pod "machines". Use the `machinectl` tool to print the list of rkt pods:

```
$ machinectl list
MACHINE                                  CLASS     SERVICE
rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23 container nspawn

1 machines listed.
```

Given the name of this rkt machine, `journalctl` can inspect its logs, or `machinectl` can shut it down:

```
# journalctl -M rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23
...
Feb 24 12:50:22 rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23 etcd[4]: 2016-02-24 11:50:22.518030 I | raft: ce2a822cea30bfca received vote from ce2a822cea30bfca at term 2
Feb 24 12:50:22 rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23 etcd[4]: 2016-02-24 11:50:22.518073 I | raft: ce2a822cea30bfca became leader at term 2
Feb 24 12:50:22 rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23 etcd[4]: 2016-02-24 11:50:22.518086 I | raft: raft.node: ce2a822cea30bfca elected leader ce2a822cea30bfca at te
Feb 24 12:50:22 rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23 etcd[4]: 2016-02-24 11:50:22.518720 I | etcdserver: published {Name:default ClientURLs:[http://localhost:2379 h
Feb 24 12:50:22 rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23 etcd[4]: 2016-02-24 11:50:22.518955 I | etcdserver: setting up the initial cluster version to 2.2
Feb 24 12:50:22 rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23 etcd[4]: 2016-02-24 11:50:22.521680 N | etcdserver: set the initial cluster version to 2.2
# machinectl poweroff rkt-2b0b2cec-8f63-4451-9431-9f8e9b265a23
$ machinectl list
MACHINE CLASS SERVICE

0 machines listed.
```

Note that journald integration is only supported if systemd is compiled with `xz` compression enabled. To inspect this, use `systemctl`:

```
$ systemctl --version
systemd v231
[...] +XZ [...]
```

If the output contains `-XZ`, journal entries will not be available.

## Managing pods as systemd services

### Notifications

systemd inside stage1 can notify systemd on the host that it is ready, to make sure that stage1 systemd send the notification at the right time you can use the [sd_notify][sd_notify] mechanism.
To make use of this feature, you need to set the annotation `appc.io/executor/supports-systemd-notify` to true in the image manifest whenever the app supports sd\_notify (see example manifest below).
If you build your image with [`acbuild`][acbuild] you can use the command: `acbuild annotation add appc.io/executor/supports-systemd-notify true`.

```
{
	"acKind": "ImageManifest",
	"acVersion": "0.8.4",
	"name": "coreos.com/etcd",
	...
	"app": {
		"exec": [
			"/etcd"
		],
		...
	},
	"annotations": [
	    "name": "appc.io/executor/supports-systemd-notify",
	    "value": "true"
	]
}
```

This feature is always available when using the "coreos" stage1 flavor.
If you use the "host" stage1 flavor (e.g. Fedora RPM or Debian deb package), you will need systemd >= v231.
To verify how it works, run in a terminal the command: `sudo systemd-run --unit=test --service-type=notify rkt run --insecure-options=image /path/to/your/app/image`, then periodically check the status with `systemctl status test`.

If the pod uses a stage1 image with systemd v231 (or greater), then the pod will be seen active form the host when systemd inside stage1 will reach default target.
Instead, before it was marked as active as soon as it started.
In this way it is possible to easily set up dependencies between pods and host services.
Moreover, using [`SdNotify()`][sdnotify-go] in the application it is possible to make the pod marked as ready when all the apps or a particular one is ready.
For more information check [systemd services unit][systemd-unit] documentation.
Below there is a simple example of an app using the systemd notification mechanism via [go-systemd][go-systemd] binding library.

```go
package main

import (
		"log"
		"net"
		"net/http"

		"github.com/coreos/go-systemd/daemon"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("request from %v\n", r.RemoteAddr)
		w.Write([]byte("hello\n"))
	})
	ln, err := net.Listen("tcp", ":5000")
	if err != nil {
		log.Fatalf("Listen failed: %s", err)
	}
	sent, err := daemon.SdNotify(true, "READY=1")
	if err != nil {
		log.Fatalf("Notification failed: %s", err)
	}
	if !sent {
		log.Fatalf("Notification not supported: %s", err)
	}
	log.Fatal(http.Serve(ln, nil))
}
```

You can run an app that supports `sd\_notify()` with this command:

```
# systemd-run --slice=machine --service-type=notify rkt run coreos.com/etcd:v2.2.5
Running as unit run-29486.service.
```

### Simple Unit File

The following is a simple example of a unit file using `rkt` to run an `etcd` instance under systemd service management:

```
[Unit]
Description=etcd

[Service]
Slice=machine.slice
ExecStart=/usr/bin/rkt run coreos.com/etcd:v2.2.5
KillMode=mixed
Restart=always
```

This unit can now be managed using the standard `systemctl` commands:

```
# systemctl start etcd.service
# systemctl stop etcd.service
# systemctl restart etcd.service
# systemctl enable etcd.service
# systemctl disable etcd.service
```

Note that no `ExecStop` clause is required. Setting [`KillMode=mixed`][systemd-killmode-mixed] means that running `systemctl stop etcd.service` will send `SIGTERM` to `stage1`'s `systemd`, which in turn will initiate orderly shutdown inside the pod. Systemd is additionally able to send the cleanup `SIGKILL` to any lingering service processes, after a timeout. This comprises complete pod lifecycle management with familiar, well-known system init tools.

### Advanced Unit File

A more advanced unit example takes advantage of a few convenient `systemd` features:

1. Inheriting environment variables specified in the unit with `--inherit-env`. This feature helps keep units concise, instead of layering on many flags to `rkt run`.
2. Using the dependency graph to start our pod after networking has come online. This is helpful if your application requires outside connectivity to fetch remote configuration (for example, from `etcd`).
3. Set resource limits for this `rkt` pod. This can also be done in the unit file, rather than flagged to `rkt run`.
4. Set `ExecStopPost` to invoke `rkt gc --mark-only` to record the timestamp when the pod exits.
(Run `rkt gc --help` to see more details about this flag).
After running `rkt gc --mark-only`, the timestamp can be retrieved from rkt API service in pod's `gc_marked_at` field.
The timestamp can be treated as the finished time of the pod.

Here is what it looks like all together:

```
[Unit]
# Metadata
Description=MyApp
Documentation=https://myapp.com/docs/1.3.4
# Wait for networking
Requires=network-online.target
After=network-online.target

[Service]
Slice=machine.slice
# Resource limits
Delegate=true
CPUShares=512
MemoryLimit=1G
# Env vars
Environment=HTTP_PROXY=192.0.2.3:5000
Environment=STORAGE_PATH=/opt/myapp
Environment=TMPDIR=/var/tmp
# Fetch the app (not strictly required, `rkt run` will fetch the image if there is not one)
ExecStartPre=/usr/bin/rkt fetch myapp.com/myapp-1.3.4
# Start the app
ExecStart=/usr/bin/rkt run --inherit-env --port=http:8888 myapp.com/myapp-1.3.4
ExecStopPost=/usr/bin/rkt gc --mark-only
KillMode=mixed
Restart=always
```

rkt must be the main process of the service in order to support [isolators][systemd-isolators] correctly and to be well-integrated with [systemd-machined][systemd-machined]. To ensure that rkt is the main process of the service, the pattern `/bin/sh -c "foo ; rkt run ..."` should be avoided, because in that case the main process is `sh`.

In most cases, the parameters `Environment=` and `ExecStartPre=` can simply be used instead of starting a shell. If shell invocation is unavoidable, use `exec` to ensure rkt replaces the preceding shell process:

```
ExecStart=/bin/sh -c "foo ; exec rkt run ..."
```

### Resource restrictions (CPU, IO, Memory)

`rkt` inherits resource limits configured in the systemd service unit file. The systemd documentation explains various [execution environment][systemd.exec], and [resource control][systemd.resource-control] settings to restrict the CPU, IO, and memory resources.

For example to restrict the CPU time quota, configure the corresponding [CPUQuota][systemd-cpuquota] setting:

```
[Service]
ExecStart=/usr/bin/rkt run s-urbaniak.github.io/images/stress:0.0.1
CPUQuota=30%
```

```
$ ps -p <PID> -o %cpu%
CPU
30.0
```

Moreover to pin the rkt pod to certain CPUs, configure the corresponding [CPUAffinity][systemd-cpuaffinity] setting:

```
[Service]
ExecStart=/usr/bin/rkt run s-urbaniak.github.io/images/stress:0.0.1
CPUAffinity=0,3
```

```
$ top
Tasks: 235 total,   1 running, 234 sleeping,   0 stopped,   0 zombie
%Cpu0  : 100.0/0.0   100[||||||||||||||||||||||||||||||||||||||||||||||
%Cpu1  :   6.0/0.7     7[|||                                           
%Cpu2  :   0.7/0.0     1[                                              
%Cpu3  : 100.0/0.0   100[||||||||||||||||||||||||||||||||||||||||||||||
GiB Mem : 25.7/19.484   [                                              
GiB Swap:  0.0/8.000    [                                              

  PID USER      PR  NI    VIRT    RES  %CPU %MEM     TIME+ S COMMAND   
11684 root      20   0    3.6m   1.1m 200.0  0.0   8:58.63 S stress    
```

### Socket-activated service

`rkt` supports [socket-activated services][systemd-socket-activated]. This means systemd will listen on a port on behalf of a container, and start the container when receiving a connection. An application needs to be able to accept sockets from systemd's native socket passing interface in order to handle socket activation.

To make socket activation work, add a [socket-activated port][aci-socketActivated] to the app container manifest:

```json
...
{
...
    "app": {
        ...
        "ports": [
            {
                "name": "80-tcp",
                "protocol": "tcp",
                "port": 80,
                "count": 1,
                "socketActivated": true
            }
        ]
    }
}
```

Then you will need a pair of `.service` and `.socket` unit files.

In this example, we want to use the port 8080 on the host instead of the app's default 80, so we use rkt's `--port` option to override it.

```
# my-socket-activated-app.socket
[Unit]
Description=My socket-activated app's socket

[Socket]
ListenStream=8080
```

```
# my-socket-activated-app.service
[Unit]
Description=My socket-activated app

[Service]
ExecStart=/usr/bin/rkt run --port 80-tcp:8080 myapp.com/my-socket-activated-app:v1.0
KillMode=mixed
```

Finally, start the socket unit:

```
# systemctl start my-socket-activated-app.socket
$ systemctl status my-socket-activated-app.socket
● my-socket-activated-app.socket - My socket-activated app's socket
   Loaded: loaded (/etc/systemd/system/my-socket-activated-app.socket; static; vendor preset: disabled)
   Active: active (listening) since Thu 2015-07-30 12:24:50 CEST; 2s ago
   Listen: [::]:8080 (Stream)

Jul 30 12:24:50 locke-work systemd[1]: Listening on My socket-activated app's socket.
```

Now, a new connection to port 8080 will start your container to handle the request.

### Bidirectionally proxy local sockets to another (possibly remote) socket.

`rkt` also supports the [socket-proxyd service][systemd-socket-proxyd]. Much like socket activation, with socket-proxyd systemd provides a listener on a given port on behalf of a container, and starts the container when a connection is received. Socket-proxy listening can be useful in environments that lack native support for socket activation. The LKVM stage1 flavor is an example of such an environment.

To set up socket proxyd, create a network template consisting of three units, like the example below. This example uses the redis app and the PTP network template in `/etc/rkt/net.d/ptp0.conf`:

```json
{
	"name": "ptp0",
	"type": "ptp",
	"ipMasq": true,
	"ipam": {
		"type": "host-local",
		"subnet": "172.16.28.0/24",
		"routes": [
			{ "dst": "0.0.0.0/0" }
		]
	}
}
```

```
# rkt-redis.service
[Unit]
Description=Socket-proxyd redis server

[Service]
ExecStart=/usr/bin/rkt --insecure-options=image run --net="ptp:IP=172.16.28.101" docker://redis
KillMode=process
```
Note that you have to specify IP manually in systemd unit.

Then you will need a pair of `.service` and `.socket` unit files.

We want to use the port 6379 on the localhost instead of the remote container IP,
so we use next systemd unit to override it.

```
# proxy-to-rkt-redis.service
[Unit]
Requires=rkt-redis.service
After=rkt-redis.service

[Service]
ExecStart=/usr/lib/systemd/systemd-socket-proxyd 172.16.28.101:6379
```
Lastly the related socket unit,
```
# proxy-to-rkt-redis.socket
[Socket]
ListenStream=6371

[Install]
WantedBy=sockets.target
```

Finally, start the socket unit:

```
# systemctl enable proxy-to-redis.socket
$ sudo systemctl start proxy-to-redis.socket
● proxy-to-rkt-redis.socket
   Loaded: loaded (/etc/systemd/system/proxy-to-rkt-redis.socket; enabled; vendor preset: disabled)
   Active: active (listening) since Mon 2016-03-07 11:53:32 CET; 8s ago
   Listen: [::]:6371 (Stream)

Mar 07 11:53:32 user-host systemd[1]: Listening on proxy-to-rkt-redis.socket.
Mar 07 11:53:32 user-host systemd[1]: Starting proxy-to-rkt-redis.socket.

```

Now, a new connection to localhost port 6371 will start your container with redis, to handle the request.

```
$ curl http://localhost:6371/
```

## Other tools for managing pods

Let us assume the service from the simple example unit file, above, is started on the host.

### ps auxf

The snippet below taken from output of `ps auxf` shows several things:

1. `rkt` `exec`s stage1's `systemd-nspawn` instead of using `fork-exec` technique. That is why rkt itself is not listed by `ps`.
2. `systemd-nspawn` runs a typical boot sequence - it spawns `systemd` inside the container, which in turn spawns our desired service(s).
3. There can be also other services running, which may be `systemd`-specific, like `systemd-journald`.

```
$ ps auxf
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root      7258  0.2  0.0  19680  2664 ?        Ss   12:38   0:02 stage1/rootfs/usr/lib/ld-linux-x86-64.so.2 stage1/rootfs/usr/bin/systemd-nspawn --boot --register=true --link-journal=try-guest --quiet --keep-unit --uuid=6d0d9608-a744-4333-be21-942145a97a5a --machine=rkt-6d0d9608-a744-4333-be21-942145a97a5a --directory=stage1/rootfs -- --default-standard-output=tty --log-target=null --log-level=warning --show-status=0
root      7275  0.0  0.0  27348  4316 ?        Ss   12:38   0:00  \_ /usr/lib/systemd/systemd --default-standard-output=tty --log-target=null --log-level=warning --show-status=0
root      7277  0.0  0.0  23832  6100 ?        Ss   12:38   0:00      \_ /usr/lib/systemd/systemd-journald
root      7343  0.3  0.0  10652  7332 ?        Ssl  12:38   0:04      \_ /etcd
```

### systemd-cgls

The `systemd-cgls` command prints the list of cgroups active on the system. The inner `system.slice` shown in the excerpt below is a cgroup in rkt's `stage1`, below which an in-container systemd has been started to shepherd pod apps with complete process lifecycle management:

```
$ systemd-cgls
├─1 /usr/lib/systemd/systemd --switched-root --system --deserialize 22
├─machine.slice
│ └─etcd.service
│   ├─1204 stage1/rootfs/usr/lib/ld-linux-x86-64.so.2 stage1/rootfs/usr/bin/s...
│   ├─1421 /usr/lib/systemd/systemd --default-standard-output=tty --log-targe...
│   └─system.slice
│     ├─etcd.service
│     │ └─1436 /etcd
│     └─systemd-journald.service
│       └─1428 /usr/lib/systemd/systemd-journald
```

### systemd-cgls --all

To display all active cgroups, use the `--all` flag. This will show two cgroups for `mount` in the host's `system.slice`. One mount cgroup is for the `stage1` root filesystem, the other for the `stage2` root (the pod's filesystem). Inside the pod's `system.slice` there are more `mount` cgroups -- mostly for bind mounts of standard `/dev`-tree device files.

```
$ systemd-cgls --all
├─1 /usr/lib/systemd/systemd --switched-root --system --deserialize 22
├─machine.slice
│ └─etcd.service
│   ├─1204 stage1/rootfs/usr/lib/ld-linux-x86-64.so.2 stage1/rootfs/usr/bin/s...
│   ├─1421 /usr/lib/systemd/systemd --default-standard-output=tty --log-targe...
│   └─system.slice
│     ├─proc-sys-kernel-random-boot_id.mount
│     ├─opt-stage2-etcd-rootfs-proc-kmsg.mount
│     ├─opt-stage2-etcd-rootfs-sys.mount
│     ├─opt-stage2-etcd-rootfs-dev-shm.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-perf_event.mount
│     ├─etcd.service
│     │ └─1436 /etcd
│     ├─opt-stage2-etcd-rootfs-proc-sys-kernel-random-boot_id.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-cpu\x2ccpuacct.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-devices.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-freezer.mount
│     ├─shutdown.service
│     ├─-.mount
│     ├─opt-stage2-etcd-rootfs-data\x2ddir.mount
│     ├─system-prepare\x2dapp.slice
│     ├─tmp.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-cpuset.mount
│     ├─opt-stage2-etcd-rootfs-proc.mount
│     ├─systemd-journald.service
│     │ └─1428 /usr/lib/systemd/systemd-journald
│     ├─opt-stage2-etcd-rootfs.mount
│     ├─opt-stage2-etcd-rootfs-dev-random.mount
│     ├─opt-stage2-etcd-rootfs-dev-pts.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup.mount
│     ├─run-systemd-nspawn-incoming.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-systemd-machine.slice-etcd.service.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-memory-machine.slice-etcd.service-system.slice-etcd.service-cgroup.procs.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-blkio.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-net_cls\x2cnet_prio.mount
│     ├─opt-stage2-etcd-rootfs-dev-net-tun.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-memory-machine.slice-etcd.service-system.slice-etcd.service-memory.limit_in_bytes.mount
│     ├─opt-stage2-etcd-rootfs-dev-tty.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-pids.mount
│     ├─reaper-etcd.service
│     ├─opt-stage2-etcd-rootfs-sys-fs-selinux.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-memory.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-cpu\x2ccpuacct-machine.slice-etcd.service-system.slice-etcd.service-cpu.cfs_quota_us.mount
│     ├─opt-stage2-etcd-rootfs-dev-urandom.mount
│     ├─opt-stage2-etcd-rootfs-dev-zero.mount
│     ├─opt-stage2-etcd-rootfs-dev-null.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-systemd.mount
│     ├─opt-stage2-etcd-rootfs-dev-console.mount
│     ├─opt-stage2-etcd-rootfs-dev-full.mount
│     ├─opt-stage2-etcd-rootfs-sys-fs-cgroup-cpu\x2ccpuacct-machine.slice-etcd.service-system.slice-etcd.service-cgroup.procs.mount
│     ├─opt-stage2-etcd-rootfs-proc-sys.mount
│     └─opt-stage2-etcd-rootfs-sys-fs-cgroup-hugetlb.mount
```


[acbuild]: https://github.com/containers/build
[aci-socketActivated]: https://github.com/appc/spec/blob/master/spec/aci.md#image-manifest-schema
[go-systemd]: https://github.com/coreos/go-systemd
[machined]: https://wiki.freedesktop.org/www/Software/systemd/machined/
[systemd]: http://www.freedesktop.org/wiki/Software/systemd/
[systemd.exec]: https://www.freedesktop.org/software/systemd/man/systemd.exec.html
[systemd.resource-control]: https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html
[systemd-cpuquota]: https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html#CPUQuota=
[systemd-cpuaffinity]: https://www.freedesktop.org/software/systemd/man/systemd.exec.html#CPUAffinity=
[systemd-isolators]: https://github.com/appc/spec/blob/master/spec/ace.md#isolators
[systemd-killmode-mixed]: http://www.freedesktop.org/software/systemd/man/systemd.kill.html#KillMode=
[systemd-machined]: http://www.freedesktop.org/software/systemd/man/systemd-machined.service.html
[systemd-run]: http://www.freedesktop.org/software/systemd/man/systemd-run.html
[systemd-socket-activated]: http://www.freedesktop.org/software/systemd/man/sd_listen_fds.html
[systemd-socket-proxyd]: https://www.freedesktop.org/software/systemd/man/systemd-socket-proxyd.html
[systemd-unit]: https://www.freedesktop.org/software/systemd/man/systemd.unit.html
[sd_notify]: https://www.freedesktop.org/software/systemd/man/sd_notify.html
[sdnotify-go]: https://github.com/coreos/go-systemd/blob/master/daemon/sdnotify.go
