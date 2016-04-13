<!--[metadata]>
+++
title = "Remote API v1.18"
description = "API Documentation for Docker"
keywords = ["API, Docker, rcli, REST,  documentation"]
[menu.main]
parent = "smn_remoteapi"
weight = 3
+++
<![end-metadata]-->

# Docker Remote API v1.18

## 1. Brief introduction

 - The Remote API has replaced `rcli`.
 - The daemon listens on `unix:///var/run/docker.sock` but you can
   [Bind Docker to another host/port or a Unix socket](
   /articles/basics/#bind-docker-to-another-hostport-or-a-unix-socket).
 - The API tends to be REST, but for some complex commands, like `attach`
   or `pull`, the HTTP connection is hijacked to transport `STDOUT`,
   `STDIN` and `STDERR`.

# 2. Endpoints

## 2.1 Containers

### List containers

`GET /containers/json`

List containers

**Example request**:

        GET /containers/json?all=1&before=8dfafdbc3a40&size=1 HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
             {
                     "Id": "8dfafdbc3a40",
                     "Names":["/boring_feynman"],
                     "Image": "ubuntu:latest",
                     "Command": "echo 1",
                     "Created": 1367854155,
                     "Status": "Exit 0",
                     "Ports": [{"PrivatePort": 2222, "PublicPort": 3333, "Type": "tcp"}],
                     "SizeRw": 12288,
                     "SizeRootFs": 0
             },
             {
                     "Id": "9cd87474be90",
                     "Names":["/coolName"],
                     "Image": "ubuntu:latest",
                     "Command": "echo 222222",
                     "Created": 1367854155,
                     "Status": "Exit 0",
                     "Ports": [],
                     "SizeRw": 12288,
                     "SizeRootFs": 0
             },
             {
                     "Id": "3176a2479c92",
                     "Names":["/sleepy_dog"],
                     "Image": "ubuntu:latest",
                     "Command": "echo 3333333333333333",
                     "Created": 1367854154,
                     "Status": "Exit 0",
                     "Ports":[],
                     "SizeRw":12288,
                     "SizeRootFs":0
             },
             {
                     "Id": "4cb07b47f9fb",
                     "Names":["/running_cat"],
                     "Image": "ubuntu:latest",
                     "Command": "echo 444444444444444444444444444444444",
                     "Created": 1367854152,
                     "Status": "Exit 0",
                     "Ports": [],
                     "SizeRw": 12288,
                     "SizeRootFs": 0
             }
        ]

Query Parameters:

-   **all** – 1/True/true or 0/False/false, Show all containers.
        Only running containers are shown by default (i.e., this defaults to false)
-   **limit** – Show `limit` last created
        containers, include non-running ones.
-   **since** – Show only containers created since Id, include
        non-running ones.
-   **before** – Show only containers created before Id, include
        non-running ones.
-   **size** – 1/True/true or 0/False/false, Show the containers
        sizes
-   **filters** - a json encoded value of the filters (a map[string][]string) to process on the containers list. Available filters:
  -   exited=&lt;int&gt; -- containers with exit code of &lt;int&gt;
  -   status=(restarting|running|paused|exited)
  -   label=`key` or `key=value` of a container label

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Create a container

`POST /containers/create`

Create a container

**Example request**:

        POST /containers/create HTTP/1.1
        Content-Type: application/json

        {
             "Hostname": "",
             "Domainname": "",
             "User": "",
             "AttachStdin": false,
             "AttachStdout": true,
             "AttachStderr": true,
             "Tty": false,
             "OpenStdin": false,
             "StdinOnce": false,
             "Env": null,
             "Cmd": [
                     "date"
             ],
             "Entrypoint": "",
             "Image": "ubuntu",
             "Labels": {
                     "com.example.vendor": "Acme",
                     "com.example.license": "GPL",
                     "com.example.version": "1.0"
             },
             "Volumes": {
                     "/tmp": {}
             },
             "WorkingDir": "",
             "NetworkDisabled": false,
             "MacAddress": "12:34:56:78:9a:bc",
             "ExposedPorts": {
                     "22/tcp": {}
             },
             "HostConfig": {
               "Binds": ["/tmp:/tmp"],
               "Links": ["redis3:redis"],
               "LxcConf": {"lxc.utsname":"docker"},
               "Memory": 0,
               "MemorySwap": 0,
               "CpuShares": 512,
               "CpusetCpus": "0,1",
               "PortBindings": { "22/tcp": [{ "HostPort": "11022" }] },
               "PublishAllPorts": false,
               "Privileged": false,
               "ReadonlyRootfs": false,
               "Dns": ["8.8.8.8"],
               "DnsSearch": [""],
               "ExtraHosts": null,
               "VolumesFrom": ["parent", "other:ro"],
               "CapAdd": ["NET_ADMIN"],
               "CapDrop": ["MKNOD"],
               "RestartPolicy": { "Name": "", "MaximumRetryCount": 0 },
               "NetworkMode": "bridge",
               "Devices": [],
               "Ulimits": [{}],
               "LogConfig": { "Type": "json-file", Config: {} },
               "SecurityOpt": [""],
               "CgroupParent": ""
            }
        }

**Example response**:

        HTTP/1.1 201 Created
        Content-Type: application/json

        {
             "Id":"e90e34656806"
             "Warnings":[]
        }

Json Parameters:

-   **Hostname** - A string value containing the desired hostname to use for the
      container.
-   **Domainname** - A string value containing the desired domain name to use
      for the container.
-   **User** - A string value containing the user to use inside the container.
-   **Memory** - Memory limit in bytes.
-   **MemorySwap**- Total memory limit (memory + swap); set `-1` to disable swap,
      always use this with `memory`, and make the value larger than `memory`.
-   **CpuShares** - An integer value containing the CPU Shares for container
      (ie. the relative weight vs other containers).
-   **Cpuset** - The same as CpusetCpus, but deprecated, please don't use.
-   **CpusetCpus** - String value containing the cgroups CpusetCpus to use.
-   **AttachStdin** - Boolean value, attaches to stdin.
-   **AttachStdout** - Boolean value, attaches to stdout.
-   **AttachStderr** - Boolean value, attaches to stderr.
-   **Tty** - Boolean value, Attach standard streams to a tty, including stdin if it is not closed.
-   **OpenStdin** - Boolean value, opens stdin,
-   **StdinOnce** - Boolean value, close stdin after the 1 attached client disconnects.
-   **Env** - A list of environment variables in the form of `VAR=value`
-   **Labels** - Adds a map of labels that to a container. To specify a map: `{"key":"value"[,"key2":"value2"]}`
-   **Cmd** - Command to run specified as a string or an array of strings.
-   **Entrypoint** - Set the entrypoint for the container a string or an array
      of strings
-   **Image** - String value containing the image name to use for the container
-   **Volumes** – An object mapping mountpoint paths (strings) inside the
      container to empty objects.
-   **WorkingDir** - A string value containing the working dir for commands to
      run in.
-   **NetworkDisabled** - Boolean value, when true disables networking for the
      container
-   **ExposedPorts** - An object mapping ports to an empty object in the form of:
      `"ExposedPorts": { "<port>/<tcp|udp>: {}" }`
-   **HostConfig**
    -   **Binds** – A list of volume bindings for this container. Each volume
            binding is a string of the form `container_path` (to create a new
            volume for the container), `host_path:container_path` (to bind-mount
            a host path into the container), or `host_path:container_path:ro`
            (to make the bind-mount read-only inside the container).
    -   **Links** - A list of links for the container. Each link entry should be
          in the form of `container_name:alias`.
    -   **LxcConf** - LXC specific configurations. These configurations will only
          work when using the `lxc` execution driver.
    -   **PortBindings** - A map of exposed container ports and the host port they
          should map to. It should be specified in the form
          `{ <port>/<protocol>: [{ "HostPort": "<port>" }] }`
          Take note that `port` is specified as a string and not an integer value.
    -   **PublishAllPorts** - Allocates a random host port for all of a container's
          exposed ports. Specified as a boolean value.
    -   **Privileged** - Gives the container full access to the host. Specified as
          a boolean value.
    -   **ReadonlyRootfs** - Mount the container's root filesystem as read only.
          Specified as a boolean value.
    -   **Dns** - A list of dns servers for the container to use.
    -   **DnsSearch** - A list of DNS search domains
    -   **ExtraHosts** - A list of hostnames/IP mappings to be added to the
        container's `/etc/hosts` file. Specified in the form `["hostname:IP"]`.
    -   **VolumesFrom** - A list of volumes to inherit from another container.
          Specified in the form `<container name>[:<ro|rw>]`
    -   **CapAdd** - A list of kernel capabilities to add to the container.
    -   **Capdrop** - A list of kernel capabilities to drop from the container.
    -   **RestartPolicy** – The behavior to apply when the container exits.  The
            value is an object with a `Name` property of either `"always"` to
            always restart or `"on-failure"` to restart only when the container
            exit code is non-zero.  If `on-failure` is used, `MaximumRetryCount`
            controls the number of times to retry before giving up.
            The default is not to restart. (optional)
            An ever increasing delay (double the previous delay, starting at 100mS)
            is added before each restart to prevent flooding the server.
    -   **NetworkMode** - Sets the networking mode for the container. Supported
          values are: `bridge`, `host`, and `container:<name|id>`
    -   **Devices** - A list of devices to add to the container specified in the
          form
          `{ "PathOnHost": "/dev/deviceName", "PathInContainer": "/dev/deviceName", "CgroupPermissions": "mrw"}`
    -   **Ulimits** - A list of ulimits to be set in the container, specified as
          `{ "Name": <name>, "Soft": <soft limit>, "Hard": <hard limit> }`, for example:
          `Ulimits: { "Name": "nofile", "Soft": 1024, "Hard", 2048 }}`
    -   **SecurityOpt**: A list of string values to customize labels for MLS
        systems, such as SELinux.
    -   **LogConfig** - Log configuration for the container, specified as
          `{ "Type": "<driver_name>", "Config": {"key1": "val1"}}`.
          Available types: `json-file`, `syslog`, `none`.
          `json-file` logging driver.
    -   **CgroupParent** - Path to cgroups under which the cgroup for the container will be created. If the path is not absolute, the path is considered to be relative to the cgroups path of the init process. Cgroups will be created if they do not already exist.

Query Parameters:

-   **name** – Assign the specified name to the container. Must
    match `/?[a-zA-Z0-9_-]+`.

Status Codes:

-   **201** – no error
-   **404** – no such container
-   **406** – impossible to attach (container not running)
-   **500** – server error

### Inspect a container

`GET /containers/(id)/json`

Return low-level information on the container `id`


**Example request**:

        GET /containers/4fa6e0f0c678/json HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

	{
		"AppArmorProfile": "",
		"Args": [
			"-c",
			"exit 9"
		],
		"Config": {
			"AttachStderr": true,
			"AttachStdin": false,
			"AttachStdout": true,
			"Cmd": [
				"/bin/sh",
				"-c",
				"exit 9"
			],
			"Domainname": "",
			"Entrypoint": null,
			"Env": [
				"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
			],
			"ExposedPorts": null,
			"Hostname": "ba033ac44011",
			"Image": "ubuntu",
			"Labels": {
				"com.example.vendor": "Acme",
				"com.example.license": "GPL",
				"com.example.version": "1.0"
			},
			"MacAddress": "",
			"NetworkDisabled": false,
			"OnBuild": null,
			"OpenStdin": false,
			"PortSpecs": null,
			"StdinOnce": false,
			"Tty": false,
			"User": "",
			"Volumes": null,
			"WorkingDir": ""
		},
		"Created": "2015-01-06T15:47:31.485331387Z",
		"Driver": "devicemapper",
		"ExecDriver": "native-0.2",
		"ExecIDs": null,
		"HostConfig": {
			"Binds": null,
			"CapAdd": null,
			"CapDrop": null,
			"ContainerIDFile": "",
			"CpusetCpus": "",
			"CpuShares": 0,
			"Devices": [],
			"Dns": null,
			"DnsSearch": null,
			"ExtraHosts": null,
			"IpcMode": "",
			"Links": null,
			"LxcConf": [],
			"Memory": 0,
			"MemorySwap": 0,
			"NetworkMode": "bridge",
			"PortBindings": {},
			"Privileged": false,
			"ReadonlyRootfs": false,
			"PublishAllPorts": false,
			"RestartPolicy": {
				"MaximumRetryCount": 2,
				"Name": "on-failure"
			},
			"LogConfig": {
				"Config": null,
				"Type": "json-file"
			},
			"SecurityOpt": null,
			"VolumesFrom": null,
			"Ulimits": [{}]
		},
		"HostnamePath": "/var/lib/docker/containers/ba033ac4401106a3b513bc9d639eee123ad78ca3616b921167cd74b20e25ed39/hostname",
		"HostsPath": "/var/lib/docker/containers/ba033ac4401106a3b513bc9d639eee123ad78ca3616b921167cd74b20e25ed39/hosts",
		"LogPath": "/var/lib/docker/containers/1eb5fabf5a03807136561b3c00adcd2992b535d624d5e18b6cdc6a6844d9767b/1eb5fabf5a03807136561b3c00adcd2992b535d624d5e18b6cdc6a6844d9767b-json.log",
		"Id": "ba033ac4401106a3b513bc9d639eee123ad78ca3616b921167cd74b20e25ed39",
		"Image": "04c5d3b7b0656168630d3ba35d8889bd0e9caafcaeb3004d2bfbc47e7c5d35d2",
		"MountLabel": "",
		"Name": "/boring_euclid",
		"NetworkSettings": {
			"Bridge": "",
			"Gateway": "",
			"IPAddress": "",
			"IPPrefixLen": 0,
			"MacAddress": "",
			"PortMapping": null,
			"Ports": null
		},
		"Path": "/bin/sh",
		"ProcessLabel": "",
		"ResolvConfPath": "/var/lib/docker/containers/ba033ac4401106a3b513bc9d639eee123ad78ca3616b921167cd74b20e25ed39/resolv.conf",
		"RestartCount": 1,
		"State": {
			"Error": "",
			"ExitCode": 9,
			"FinishedAt": "2015-01-06T15:47:32.080254511Z",
			"OOMKilled": false,
			"Paused": false,
			"Pid": 0,
			"Restarting": false,
			"Running": false,
			"StartedAt": "2015-01-06T15:47:32.072697474Z"
		},
		"Volumes": {},
		"VolumesRW": {}
	}

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

### List processes running inside a container

`GET /containers/(id)/top`

List processes running inside the container `id`

**Example request**:

        GET /containers/4fa6e0f0c678/top HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
             "Titles": [
                     "USER",
                     "PID",
                     "%CPU",
                     "%MEM",
                     "VSZ",
                     "RSS",
                     "TTY",
                     "STAT",
                     "START",
                     "TIME",
                     "COMMAND"
                     ],
             "Processes": [
                     ["root","20147","0.0","0.1","18060","1864","pts/4","S","10:06","0:00","bash"],
                     ["root","20271","0.0","0.0","4312","352","pts/4","S+","10:07","0:00","sleep","10"]
             ]
        }

Query Parameters:

-   **ps_args** – ps arguments to use (e.g., aux)

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

### Get container logs

`GET /containers/(id)/logs`

Get stdout and stderr logs from the container ``id``

> **Note**:
> This endpoint works only for containers with `json-file` logging driver.

**Example request**:

       GET /containers/4fa6e0f0c678/logs?stderr=1&stdout=1&timestamps=1&follow=1&tail=10 HTTP/1.1

**Example response**:

       HTTP/1.1 101 UPGRADED
       Content-Type: application/vnd.docker.raw-stream
       Connection: Upgrade
       Upgrade: tcp

       {{ STREAM }}

Query Parameters:

-   **follow** – 1/True/true or 0/False/false, return stream. Default false
-   **stdout** – 1/True/true or 0/False/false, show stdout log. Default false
-   **stderr** – 1/True/true or 0/False/false, show stderr log. Default false
-   **timestamps** – 1/True/true or 0/False/false, print timestamps for
        every log line. Default false
-   **tail** – Output specified number of lines at the end of logs: `all` or `<number>`. Default all

Status Codes:

-   **101** – no error, hints proxy about hijacking
-   **200** – no error, no upgrade header found
-   **404** – no such container
-   **500** – server error

### Inspect changes on a container's filesystem

`GET /containers/(id)/changes`

Inspect changes on container `id`'s filesystem

**Example request**:

        GET /containers/4fa6e0f0c678/changes HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
             {
                     "Path": "/dev",
                     "Kind": 0
             },
             {
                     "Path": "/dev/kmsg",
                     "Kind": 1
             },
             {
                     "Path": "/test",
                     "Kind": 1
             }
        ]

Values for `Kind`:

- `0`: Modify
- `1`: Add
- `2`: Delete

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

### Export a container

`GET /containers/(id)/export`

Export the contents of container `id`

**Example request**:

        GET /containers/4fa6e0f0c678/export HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/octet-stream

        {{ TAR STREAM }}

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

### Get container stats based on resource usage

`GET /containers/(id)/stats`

This endpoint returns a live stream of a container's resource usage statistics.

> **Note**: this functionality currently only works when using the *libcontainer* exec-driver.

**Example request**:

        GET /containers/redis1/stats HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
           "read" : "2015-01-08T22:57:31.547920715Z",
           "network" : {
              "rx_dropped" : 0,
              "rx_bytes" : 648,
              "rx_errors" : 0,
              "tx_packets" : 8,
              "tx_dropped" : 0,
              "rx_packets" : 8,
              "tx_errors" : 0,
              "tx_bytes" : 648
           },
           "memory_stats" : {
              "stats" : {
                 "total_pgmajfault" : 0,
                 "cache" : 0,
                 "mapped_file" : 0,
                 "total_inactive_file" : 0,
                 "pgpgout" : 414,
                 "rss" : 6537216,
                 "total_mapped_file" : 0,
                 "writeback" : 0,
                 "unevictable" : 0,
                 "pgpgin" : 477,
                 "total_unevictable" : 0,
                 "pgmajfault" : 0,
                 "total_rss" : 6537216,
                 "total_rss_huge" : 6291456,
                 "total_writeback" : 0,
                 "total_inactive_anon" : 0,
                 "rss_huge" : 6291456,
                 "hierarchical_memory_limit" : 67108864,
                 "total_pgfault" : 964,
                 "total_active_file" : 0,
                 "active_anon" : 6537216,
                 "total_active_anon" : 6537216,
                 "total_pgpgout" : 414,
                 "total_cache" : 0,
                 "inactive_anon" : 0,
                 "active_file" : 0,
                 "pgfault" : 964,
                 "inactive_file" : 0,
                 "total_pgpgin" : 477
              },
              "max_usage" : 6651904,
              "usage" : 6537216,
              "failcnt" : 0,
              "limit" : 67108864
           },
           "blkio_stats" : {},
           "cpu_stats" : {
              "cpu_usage" : {
                 "percpu_usage" : [
                    16970827,
                    1839451,
                    7107380,
                    10571290
                 ],
                 "usage_in_usermode" : 10000000,
                 "total_usage" : 36488948,
                 "usage_in_kernelmode" : 20000000
              },
              "system_cpu_usage" : 20091722000000000,
              "throttling_data" : {}
           }
        }

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

### Resize a container TTY

`POST /containers/(id)/resize?h=<height>&w=<width>`

Resize the TTY for container with  `id`. The container must be restarted for the resize to take effect.

**Example request**:

        POST /containers/4fa6e0f0c678/resize?h=40&w=80 HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Length: 0
        Content-Type: text/plain; charset=utf-8

Status Codes:

-   **200** – no error
-   **404** – No such container
-   **500** – Cannot resize container

### Start a container

`POST /containers/(id)/start`

Start the container `id`

> **Note**:
> For backwards compatibility, this endpoint accepts a `HostConfig` as JSON-encoded request body.
> See [create a container](#create-a-container) for details.

**Example request**:

        POST /containers/(id)/start HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Status Codes:

-   **204** – no error
-   **304** – container already started
-   **404** – no such container
-   **500** – server error

### Stop a container

`POST /containers/(id)/stop`

Stop the container `id`

**Example request**:

        POST /containers/e90e34656806/stop?t=5 HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Query Parameters:

-   **t** – number of seconds to wait before killing the container

Status Codes:

-   **204** – no error
-   **304** – container already stopped
-   **404** – no such container
-   **500** – server error

### Restart a container

`POST /containers/(id)/restart`

Restart the container `id`

**Example request**:

        POST /containers/e90e34656806/restart?t=5 HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Query Parameters:

-   **t** – number of seconds to wait before killing the container

Status Codes:

-   **204** – no error
-   **404** – no such container
-   **500** – server error

### Kill a container

`POST /containers/(id)/kill`

Kill the container `id`

**Example request**:

        POST /containers/e90e34656806/kill HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Query Parameters

-   **signal** - Signal to send to the container: integer or string like "SIGINT".
        When not set, SIGKILL is assumed and the call will waits for the container to exit.

Status Codes:

-   **204** – no error
-   **404** – no such container
-   **500** – server error

### Rename a container

`POST /containers/(id)/rename`

Rename the container `id` to a `new_name`

**Example request**:

        POST /containers/e90e34656806/rename?name=new_name HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Query Parameters:

-   **name** – new name for the container

Status Codes:

-   **204** – no error
-   **404** – no such container
-   **409** - conflict name already assigned
-   **500** – server error

### Pause a container

`POST /containers/(id)/pause`

Pause the container `id`

**Example request**:

        POST /containers/e90e34656806/pause HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Status Codes:

-   **204** – no error
-   **404** – no such container
-   **500** – server error

### Unpause a container

`POST /containers/(id)/unpause`

Unpause the container `id`

**Example request**:

        POST /containers/e90e34656806/unpause HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Status Codes:

-   **204** – no error
-   **404** – no such container
-   **500** – server error

### Attach to a container

`POST /containers/(id)/attach`

Attach to the container `id`

**Example request**:

        POST /containers/16253994b7c4/attach?logs=1&stream=0&stdout=1 HTTP/1.1

**Example response**:

        HTTP/1.1 101 UPGRADED
        Content-Type: application/vnd.docker.raw-stream
        Connection: Upgrade
        Upgrade: tcp

        {{ STREAM }}

Query Parameters:

-   **logs** – 1/True/true or 0/False/false, return logs. Default false
-   **stream** – 1/True/true or 0/False/false, return stream.
        Default false
-   **stdin** – 1/True/true or 0/False/false, if stream=true, attach
        to stdin. Default false
-   **stdout** – 1/True/true or 0/False/false, if logs=true, return
        stdout log, if stream=true, attach to stdout. Default false
-   **stderr** – 1/True/true or 0/False/false, if logs=true, return
        stderr log, if stream=true, attach to stderr. Default false

Status Codes:

-   **101** – no error, hints proxy about hijacking
-   **200** – no error, no upgrade header found
-   **400** – bad parameter
-   **404** – no such container
-   **500** – server error

    **Stream details**:

    When using the TTY setting is enabled in
    [`POST /containers/create`
    ](/reference/api/docker_remote_api_v1.9/#create-a-container "POST /containers/create"),
    the stream is the raw data from the process PTY and client's stdin.
    When the TTY is disabled, then the stream is multiplexed to separate
    stdout and stderr.

    The format is a **Header** and a **Payload** (frame).

    **HEADER**

    The header will contain the information on which stream write the
    stream (stdout or stderr). It also contain the size of the
    associated frame encoded on the last 4 bytes (uint32).

    It is encoded on the first 8 bytes like this:

        header := [8]byte{STREAM_TYPE, 0, 0, 0, SIZE1, SIZE2, SIZE3, SIZE4}

    `STREAM_TYPE` can be:

-   0: stdin (will be written on stdout)
-   1: stdout
-   2: stderr

    `SIZE1, SIZE2, SIZE3, SIZE4` are the 4 bytes of
    the uint32 size encoded as big endian.

    **PAYLOAD**

    The payload is the raw stream.

    **IMPLEMENTATION**

    The simplest way to implement the Attach protocol is the following:

    1.  Read 8 bytes
    2.  chose stdout or stderr depending on the first byte
    3.  Extract the frame size from the last 4 bytes
    4.  Read the extracted size and output it on the correct output
    5.  Goto 1

### Attach to a container (websocket)

`GET /containers/(id)/attach/ws`

Attach to the container `id` via websocket

Implements websocket protocol handshake according to [RFC 6455](http://tools.ietf.org/html/rfc6455)

**Example request**

        GET /containers/e90e34656806/attach/ws?logs=0&stream=1&stdin=1&stdout=1&stderr=1 HTTP/1.1

**Example response**

        {{ STREAM }}

Query Parameters:

-   **logs** – 1/True/true or 0/False/false, return logs. Default false
-   **stream** – 1/True/true or 0/False/false, return stream.
        Default false
-   **stdin** – 1/True/true or 0/False/false, if stream=true, attach
        to stdin. Default false
-   **stdout** – 1/True/true or 0/False/false, if logs=true, return
        stdout log, if stream=true, attach to stdout. Default false
-   **stderr** – 1/True/true or 0/False/false, if logs=true, return
        stderr log, if stream=true, attach to stderr. Default false

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **404** – no such container
-   **500** – server error

### Wait a container

`POST /containers/(id)/wait`

Block until container `id` stops, then returns the exit code

**Example request**:

        POST /containers/16253994b7c4/wait HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"StatusCode": 0}

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

### Remove a container

`DELETE /containers/(id)`

Remove the container `id` from the filesystem

**Example request**:

        DELETE /containers/16253994b7c4?v=1 HTTP/1.1

**Example response**:

        HTTP/1.1 204 No Content

Query Parameters:

-   **v** – 1/True/true or 0/False/false, Remove the volumes
        associated to the container. Default false
-   **force** - 1/True/true or 0/False/false, Kill then remove the container.
        Default false

Status Codes:

-   **204** – no error
-   **400** – bad parameter
-   **404** – no such container
-   **500** – server error

### Copy files or folders from a container

`POST /containers/(id)/copy`

Copy files or folders of container `id`

**Example request**:

        POST /containers/4fa6e0f0c678/copy HTTP/1.1
        Content-Type: application/json

        {
             "Resource": "test.txt"
        }

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/x-tar

        {{ TAR STREAM }}

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

## 2.2 Images

### List Images

`GET /images/json`

**Example request**:

        GET /images/json?all=0 HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
          {
             "RepoTags": [
               "ubuntu:12.04",
               "ubuntu:precise",
               "ubuntu:latest"
             ],
             "Id": "8dbd9e392a964056420e5d58ca5cc376ef18e2de93b5cc90e868a1bbc8318c1c",
             "Created": 1365714795,
             "Size": 131506275,
             "VirtualSize": 131506275
          },
          {
             "RepoTags": [
               "ubuntu:12.10",
               "ubuntu:quantal"
             ],
             "ParentId": "27cf784147099545",
             "Id": "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
             "Created": 1364102658,
             "Size": 24653,
             "VirtualSize": 180116135
          }
        ]

**Example request, with digest information**:

        GET /images/json?digests=1 HTTP/1.1

**Example response, with digest information**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
          {
            "Created": 1420064636,
            "Id": "4986bf8c15363d1c5d15512d5266f8777bfba4974ac56e3270e7760f6f0a8125",
            "ParentId": "ea13149945cb6b1e746bf28032f02e9b5a793523481a0a18645fc77ad53c4ea2",
            "RepoDigests": [
              "localhost:5000/test/busybox@sha256:cbbf2f9a99b47fc460d422812b6a5adff7dfee951d8fa2e4a98caa0382cfbdbf"
            ],
            "RepoTags": [
              "localhost:5000/test/busybox:latest",
              "playdate:latest"
            ],
            "Size": 0,
            "VirtualSize": 2429728
          }
        ]

The response shows a single image `Id` associated with two repositories
(`RepoTags`): `localhost:5000/test/busybox`: and `playdate`. A caller can use
either of the `RepoTags` values `localhost:5000/test/busybox:latest` or
`playdate:latest` to reference the image.

You can also use `RepoDigests` values to reference an image. In this response,
the array has only one reference and that is to the
`localhost:5000/test/busybox` repository; the `playdate` repository has no
digest. You can reference this digest using the value:
`localhost:5000/test/busybox@sha256:cbbf2f9a99b47fc460d...`

See the `docker run` and `docker build` commands for examples of digest and tag
references on the command line.

Query Parameters:

-   **all** – 1/True/true or 0/False/false, default false
-   **filters** – a json encoded value of the filters (a map[string][]string) to process on the images list. Available filters:
  -   dangling=true
  -   label=`key` or `key=value` of an image label

### Build image from a Dockerfile

`POST /build`

Build an image from a Dockerfile

**Example request**:

        POST /build HTTP/1.1

        {{ TAR STREAM }}

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"stream": "Step 1..."}
        {"stream": "..."}
        {"error": "Error...", "errorDetail": {"code": 123, "message": "Error..."}}

The input stream must be a tar archive compressed with one of the
following algorithms: identity (no compression), gzip, bzip2, xz.

The archive must include a build instructions file, typically called
`Dockerfile` at the root of the archive. The `dockerfile` parameter may be
used to specify a different build instructions file by having its value be
the path to the alternate build instructions file to use.

The archive may include any number of other files,
which will be accessible in the build context (See the [*ADD build
command*](/reference/builder/#dockerbuilder)).

The build will also be canceled if the client drops the connection by quitting
or being killed.

Query Parameters:

-   **dockerfile** - path within the build context to the Dockerfile. This is 
        ignored if `remote` is specified and points to an individual filename.
-   **t** – repository name (and optionally a tag) to be applied to
        the resulting image in case of success
-   **remote** – A Git repository URI or HTTP/HTTPS URI build source. If the 
        URI specifies a filename, the file's contents are placed into a file 
		called `Dockerfile`.
-   **q** – suppress verbose build output
-   **nocache** – do not use the cache when building the image
-   **pull** - attempt to pull the image even if an older image exists locally
-   **rm** - remove intermediate containers after a successful build (default behavior)
-   **forcerm** - always remove intermediate containers (includes rm)
-   **memory** - set memory limit for build
-   **memswap** - Total memory (memory + swap), `-1` to disable swap
-   **cpushares** - CPU shares (relative weight)
-   **cpusetcpus** - CPUs in which to allow execution, e.g., `0-3`, `0,1`

    Request Headers:

-   **Content-type** – should be set to `"application/tar"`.
-   **X-Registry-Config** – base64-encoded ConfigFile object

Status Codes:

-   **200** – no error
-   **500** – server error

### Create an image

`POST /images/create`

Create an image, either by pulling it from the registry or by importing it

**Example request**:

        POST /images/create?fromImage=ubuntu HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"status": "Pulling..."}
        {"status": "Pulling", "progress": "1 B/ 100 B", "progressDetail": {"current": 1, "total": 100}}
        {"error": "Invalid..."}
        ...

    When using this endpoint to pull an image from the registry, the
    `X-Registry-Auth` header can be used to include
    a base64-encoded AuthConfig object.

Query Parameters:

-   **fromImage** – name of the image to pull
-   **fromSrc** – source to import.  The value may be a URL from which the image
        can be retrieved or `-` to read the image from the request body.
-   **repo** – repository
-   **tag** – tag
-   **registry** – the registry to pull from

    Request Headers:

-   **X-Registry-Auth** – base64-encoded AuthConfig object

Status Codes:

-   **200** – no error
-   **500** – server error



### Inspect an image

`GET /images/(name)/json`

Return low-level information on the image `name`

**Example request**:

        GET /images/ubuntu/json HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
             "Created": "2013-03-23T22:24:18.818426-07:00",
             "Container": "3d67245a8d72ecf13f33dffac9f79dcdf70f75acb84d308770391510e0c23ad0",
             "ContainerConfig":
                     {
                             "Hostname": "",
                             "User": "",
                             "AttachStdin": false,
                             "AttachStdout": false,
                             "AttachStderr": false,
                             "PortSpecs": null,
                             "Tty": true,
                             "OpenStdin": true,
                             "StdinOnce": false,
                             "Env": null,
                             "Cmd": ["/bin/bash"],
                             "Dns": null,
                             "Image": "ubuntu",
                             "Labels": {
                                 "com.example.vendor": "Acme",
                                 "com.example.license": "GPL",
                                 "com.example.version": "1.0"
                             },
                             "Volumes": null,
                             "VolumesFrom": "",
                             "WorkingDir": ""
                     },
             "Id": "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
             "Parent": "27cf784147099545",
             "Size": 6824592
        }

Status Codes:

-   **200** – no error
-   **404** – no such image
-   **500** – server error

### Get the history of an image

`GET /images/(name)/history`

Return the history of the image `name`

**Example request**:

        GET /images/ubuntu/history HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
             {
                     "Id": "b750fe79269d",
                     "Created": 1364102658,
                     "CreatedBy": "/bin/bash"
             },
             {
                     "Id": "27cf78414709",
                     "Created": 1364068391,
                     "CreatedBy": ""
             }
        ]

Status Codes:

-   **200** – no error
-   **404** – no such image
-   **500** – server error

### Push an image on the registry

`POST /images/(name)/push`

Push the image `name` on the registry

**Example request**:

        POST /images/test/push HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"status": "Pushing..."}
        {"status": "Pushing", "progress": "1/? (n/a)", "progressDetail": {"current": 1}}}
        {"error": "Invalid..."}
        ...

    If you wish to push an image on to a private registry, that image must already have been tagged
    into a repository which references that registry host name and port.  This repository name should
    then be used in the URL. This mirrors the flow of the CLI.

**Example request**:

        POST /images/registry.acme.com:5000/test/push HTTP/1.1


Query Parameters:

-   **tag** – the tag to associate with the image on the registry, optional

Request Headers:

-   **X-Registry-Auth** – include a base64-encoded AuthConfig
        object.

Status Codes:

-   **200** – no error
-   **404** – no such image
-   **500** – server error

### Tag an image into a repository

`POST /images/(name)/tag`

Tag the image `name` into a repository

**Example request**:

        POST /images/test/tag?repo=myrepo&force=0&tag=v42 HTTP/1.1

**Example response**:

        HTTP/1.1 201 OK

Query Parameters:

-   **repo** – The repository to tag in
-   **force** – 1/True/true or 0/False/false, default false
-   **tag** - The new tag name

Status Codes:

-   **201** – no error
-   **400** – bad parameter
-   **404** – no such image
-   **409** – conflict
-   **500** – server error

### Remove an image

`DELETE /images/(name)`

Remove the image `name` from the filesystem

**Example request**:

        DELETE /images/test HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-type: application/json

        [
         {"Untagged": "3e2f21a89f"},
         {"Deleted": "3e2f21a89f"},
         {"Deleted": "53b4f83ac9"}
        ]

Query Parameters:

-   **force** – 1/True/true or 0/False/false, default false
-   **noprune** – 1/True/true or 0/False/false, default false

Status Codes:

-   **200** – no error
-   **404** – no such image
-   **409** – conflict
-   **500** – server error

### Search images

`GET /images/search`

Search for an image on [Docker Hub](https://hub.docker.com).

> **Note**:
> The response keys have changed from API v1.6 to reflect the JSON
> sent by the registry server to the docker daemon's request.

**Example request**:

        GET /images/search?term=sshd HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
                {
                    "description": "",
                    "is_official": false,
                    "is_automated": false,
                    "name": "wma55/u1210sshd",
                    "star_count": 0
                },
                {
                    "description": "",
                    "is_official": false,
                    "is_automated": false,
                    "name": "jdswinbank/sshd",
                    "star_count": 0
                },
                {
                    "description": "",
                    "is_official": false,
                    "is_automated": false,
                    "name": "vgauthier/sshd",
                    "star_count": 0
                }
        ...
        ]

Query Parameters:

-   **term** – term to search

Status Codes:

-   **200** – no error
-   **500** – server error

## 2.3 Misc

### Check auth configuration

`POST /auth`

Get the default username and email

**Example request**:

        POST /auth HTTP/1.1
        Content-Type: application/json

        {
             "username":" hannibal",
             "password: "xxxx",
             "email": "hannibal@a-team.com",
             "serveraddress": "https://index.docker.io/v1/"
        }

**Example response**:

        HTTP/1.1 200 OK

Status Codes:

-   **200** – no error
-   **204** – no error
-   **500** – server error

### Display system-wide information

`GET /info`

Display system-wide information

**Example request**:

        GET /info HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
            "Containers": 11,
            "Debug": 0,
            "DockerRootDir": "/var/lib/docker",
            "Driver": "btrfs",
            "DriverStatus": [[""]],
            "ExecutionDriver": "native-0.1",
            "HttpProxy": "http://test:test@localhost:8080",
            "HttpsProxy": "https://test:test@localhost:8080",
            "ID": "7TRN:IPZB:QYBB:VPBQ:UMPP:KARE:6ZNR:XE6T:7EWV:PKF4:ZOJD:TPYS",
            "IPv4Forwarding": 1,
            "Images": 16,
            "IndexServerAddress": "https://index.docker.io/v1/",
            "InitPath": "/usr/bin/docker",
            "InitSha1": "",
            "KernelVersion": "3.12.0-1-amd64",
            "Labels": [
                "storage=ssd"
            ],
            "MemTotal": 2099236864,
            "MemoryLimit": 1,
            "NCPU": 1,
            "NEventsListener": 0,
            "NFd": 11,
            "NGoroutines": 21,
            "Name": "prod-server-42",
            "NoProxy": "9.81.1.160",
            "OperatingSystem": "Boot2Docker",
            "RegistryConfig": {
                "IndexConfigs": {
                    "docker.io": {
                        "Mirrors": null,
                        "Name": "docker.io",
                        "Official": true,
                        "Secure": true
                    }
                },
                "InsecureRegistryCIDRs": [
                    "127.0.0.0/8"
                ]
            },
            "SwapLimit": 0,
            "SystemTime": "2015-03-10T11:11:23.730591467-07:00"
        }
        
Status Codes:

-   **200** – no error
-   **500** – server error

### Show the docker version information

`GET /version`

Show the docker version information

**Example request**:

        GET /version HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
             "Version": "1.5.0",
             "Os": "linux",
             "KernelVersion": "3.18.5-tinycore64",
             "GoVersion": "go1.4.1",
             "GitCommit": "a8a31ef",
             "Arch": "amd64",
             "ApiVersion": "1.18"
        }

Status Codes:

-   **200** – no error
-   **500** – server error

### Ping the docker server

`GET /_ping`

Ping the docker server

**Example request**:

        GET /_ping HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: text/plain

        OK

Status Codes:

-   **200** - no error
-   **500** - server error

### Create a new image from a container's changes

`POST /commit`

Create a new image from a container's changes

**Example request**:

        POST /commit?container=44c004db4b17&comment=message&repo=myrepo HTTP/1.1
        Content-Type: application/json

        {
             "Hostname": "",
             "Domainname": "",
             "User": "",
             "AttachStdin": false,
             "AttachStdout": true,
             "AttachStderr": true,
             "PortSpecs": null,
             "Tty": false,
             "OpenStdin": false,
             "StdinOnce": false,
             "Env": null,
             "Cmd": [
                     "date"
             ],
             "Volumes": {
                     "/tmp": {}
             },
             "WorkingDir": "",
             "NetworkDisabled": false,
             "ExposedPorts": {
                     "22/tcp": {}
             }
        }

**Example response**:

        HTTP/1.1 201 Created
        Content-Type: application/vnd.docker.raw-stream

        {"Id": "596069db4bf5"}

Json Parameters:

-  **config** - the container's configuration

Query Parameters:

-   **container** – source container
-   **repo** – repository
-   **tag** – tag
-   **comment** – commit message
-   **author** – author (e.g., "John Hannibal Smith
    <[hannibal@a-team.com](mailto:hannibal%40a-team.com)>")

Status Codes:

-   **201** – no error
-   **404** – no such container
-   **500** – server error

### Monitor Docker's events

`GET /events`

Get container events from docker, either in real time via streaming, or via
polling (using since).

Docker containers will report the following events:

    create, destroy, die, exec_create, exec_start, export, kill, oom, pause, restart, start, stop, unpause

and Docker images will report:

    untag, delete

**Example request**:

        GET /events?since=1374067924

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"status": "create", "id": "dfdf82bd3881","from": "ubuntu:latest", "time":1374067924}
        {"status": "start", "id": "dfdf82bd3881","from": "ubuntu:latest", "time":1374067924}
        {"status": "stop", "id": "dfdf82bd3881","from": "ubuntu:latest", "time":1374067966}
        {"status": "destroy", "id": "dfdf82bd3881","from": "ubuntu:latest", "time":1374067970}

Query Parameters:

-   **since** – timestamp used for polling
-   **until** – timestamp used for polling
-   **filters** – a json encoded value of the filters (a map[string][]string) to process on the event list. Available filters:
  -   event=&lt;string&gt; -- event to filter
  -   image=&lt;string&gt; -- image to filter
  -   container=&lt;string&gt; -- container to filter

Status Codes:

-   **200** – no error
-   **500** – server error

### Get a tarball containing all images in a repository

`GET /images/(name)/get`

Get a tarball containing all images and metadata for the repository specified
by `name`.

If `name` is a specific name and tag (e.g. ubuntu:latest), then only that image
(and its parents) are returned. If `name` is an image ID, similarly only that
image (and its parents) are returned, but with the exclusion of the
'repositories' file in the tarball, as there were no image names referenced.

See the [image tarball format](#image-tarball-format) for more details.

**Example request**

        GET /images/ubuntu/get

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/x-tar

        Binary data stream

Status Codes:

-   **200** – no error
-   **500** – server error

### Get a tarball containing all images.

`GET /images/get`

Get a tarball containing all images and metadata for one or more repositories.

For each value of the `names` parameter: if it is a specific name and tag (e.g.
ubuntu:latest), then only that image (and its parents) are returned; if it is
an image ID, similarly only that image (and its parents) are returned and there
would be no names referenced in the 'repositories' file for this image ID.

See the [image tarball format](#image-tarball-format) for more details.

**Example request**

        GET /images/get?names=myname%2Fmyapp%3Alatest&names=busybox

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/x-tar

        Binary data stream

Status Codes:

-   **200** – no error
-   **500** – server error

### Load a tarball with a set of images and tags into docker

`POST /images/load`

Load a set of images and tags into the docker repository.
See the [image tarball format](#image-tarball-format) for more details.

**Example request**

        POST /images/load

        Tarball in body

**Example response**:

        HTTP/1.1 200 OK

Status Codes:

-   **200** – no error
-   **500** – server error

### Image tarball format

An image tarball contains one directory per image layer (named using its long ID),
each containing three files:

1. `VERSION`: currently `1.0` - the file format version
2. `json`: detailed layer information, similar to `docker inspect layer_id`
3. `layer.tar`: A tarfile containing the filesystem changes in this layer

The `layer.tar` file will contain `aufs` style `.wh..wh.aufs` files and directories
for storing attribute changes and deletions.

If the tarball defines a repository, there will also be a `repositories` file at
the root that contains a list of repository and tag names mapped to layer IDs.

```
{"hello-world":
    {"latest": "565a9d68a73f6706862bfe8409a7f659776d4d60a8d096eb4a3cbce6999cc2a1"}
}
```

### Exec Create

`POST /containers/(id)/exec`

Sets up an exec instance in a running container `id`

**Example request**:

        POST /containers/e90e34656806/exec HTTP/1.1
        Content-Type: application/json

        {
	     "AttachStdin": false,
	     "AttachStdout": true,
	     "AttachStderr": true,
	     "Tty": false,
	     "Cmd": [
                     "date"
             ],
        }

**Example response**:

        HTTP/1.1 201 OK
        Content-Type: application/json

        {
             "Id": "f90e34656806"
             "Warnings":[]
        }

Json Parameters:

-   **AttachStdin** - Boolean value, attaches to stdin of the exec command.
-   **AttachStdout** - Boolean value, attaches to stdout of the exec command.
-   **AttachStderr** - Boolean value, attaches to stderr of the exec command.
-   **Tty** - Boolean value to allocate a pseudo-TTY
-   **Cmd** - Command to run specified as a string or an array of strings.


Status Codes:

-   **201** – no error
-   **404** – no such container

### Exec Start

`POST /exec/(id)/start`

Starts a previously set up exec instance `id`. If `detach` is true, this API
returns after starting the `exec` command. Otherwise, this API sets up an
interactive session with the `exec` command.

**Example request**:

        POST /exec/e90e34656806/start HTTP/1.1
        Content-Type: application/json

        {
	     "Detach": false,
	     "Tty": false,
        }

**Example response**:

        HTTP/1.1 201 OK
        Content-Type: application/json

        {{ STREAM }}

Json Parameters:

-   **Detach** - Detach from the exec command
-   **Tty** - Boolean value to allocate a pseudo-TTY

Status Codes:

-   **201** – no error
-   **404** – no such exec instance

    **Stream details**:
    Similar to the stream behavior of `POST /container/(id)/attach` API

### Exec Resize

`POST /exec/(id)/resize`

Resizes the tty session used by the exec command `id`.
This API is valid only if `tty` was specified as part of creating and starting the exec command.

**Example request**:

        POST /exec/e90e34656806/resize HTTP/1.1
        Content-Type: text/plain

**Example response**:

        HTTP/1.1 201 OK
        Content-Type: text/plain

Query Parameters:

-   **h** – height of tty session
-   **w** – width

Status Codes:

-   **201** – no error
-   **404** – no such exec instance

### Exec Inspect

`GET /exec/(id)/json`

Return low-level information about the exec command `id`.

**Example request**:

        GET /exec/11fb006128e8ceb3942e7c58d77750f24210e35f879dd204ac975c184b820b39/json HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: plain/text

        {
          "ID" : "11fb006128e8ceb3942e7c58d77750f24210e35f879dd204ac975c184b820b39",
          "Running" : false,
          "ExitCode" : 2,
          "ProcessConfig" : {
            "privileged" : false,
            "user" : "",
            "tty" : false,
            "entrypoint" : "sh",
            "arguments" : [
              "-c",
              "exit 2"
            ]
          },
          "OpenStdin" : false,
          "OpenStderr" : false,
          "OpenStdout" : false,
          "Container" : {
            "State" : {
              "Running" : true,
              "Paused" : false,
              "Restarting" : false,
              "OOMKilled" : false,
              "Pid" : 3650,
              "ExitCode" : 0,
              "Error" : "",
              "StartedAt" : "2014-11-17T22:26:03.717657531Z",
              "FinishedAt" : "0001-01-01T00:00:00Z"
            },
            "ID" : "8f177a186b977fb451136e0fdf182abff5599a08b3c7f6ef0d36a55aaf89634c",
            "Created" : "2014-11-17T22:26:03.626304998Z",
            "Path" : "date",
            "Args" : [],
            "Config" : {
              "Hostname" : "8f177a186b97",
              "Domainname" : "",
              "User" : "",
              "AttachStdin" : false,
              "AttachStdout" : false,
              "AttachStderr" : false,
              "PortSpecs" : null,
              "ExposedPorts" : null,
              "Tty" : false,
              "OpenStdin" : false,
              "StdinOnce" : false,
              "Env" : [ "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" ],
              "Cmd" : [
                "date"
              ],
              "Image" : "ubuntu",
              "Volumes" : null,
              "WorkingDir" : "",
              "Entrypoint" : null,
              "NetworkDisabled" : false,
              "MacAddress" : "",
              "OnBuild" : null,
              "SecurityOpt" : null
            },
            "Image" : "5506de2b643be1e6febbf3b8a240760c6843244c41e12aa2f60ccbb7153d17f5",
            "NetworkSettings" : {
              "IPAddress" : "172.17.0.2",
              "IPPrefixLen" : 16,
              "MacAddress" : "02:42:ac:11:00:02",
              "Gateway" : "172.17.42.1",
              "Bridge" : "docker0",
              "PortMapping" : null,
              "Ports" : {}
            },
            "ResolvConfPath" : "/var/lib/docker/containers/8f177a186b977fb451136e0fdf182abff5599a08b3c7f6ef0d36a55aaf89634c/resolv.conf",
            "HostnamePath" : "/var/lib/docker/containers/8f177a186b977fb451136e0fdf182abff5599a08b3c7f6ef0d36a55aaf89634c/hostname",
            "HostsPath" : "/var/lib/docker/containers/8f177a186b977fb451136e0fdf182abff5599a08b3c7f6ef0d36a55aaf89634c/hosts",
            "LogPath": "/var/lib/docker/containers/1eb5fabf5a03807136561b3c00adcd2992b535d624d5e18b6cdc6a6844d9767b/1eb5fabf5a03807136561b3c00adcd2992b535d624d5e18b6cdc6a6844d9767b-json.log",
            "Name" : "/test",
            "Driver" : "aufs",
            "ExecDriver" : "native-0.2",
            "MountLabel" : "",
            "ProcessLabel" : "",
            "AppArmorProfile" : "",
            "RestartCount" : 0,
            "Volumes" : {},
            "VolumesRW" : {}
          }
        }

Status Codes:

-   **200** – no error
-   **404** – no such exec instance
-   **500** - server error

# 3. Going further

## 3.1 Inside `docker run`

As an example, the `docker run` command line makes the following API calls:

- Create the container

- If the status code is 404, it means the image doesn't exist:
    - Try to pull it
    - Then retry to create the container

- Start the container

- If you are not in detached mode:
- Attach to the container, using logs=1 (to have stdout and
      stderr from the container's start) and stream=1

- If in detached mode or only stdin is attached:
- Display the container's id

## 3.2 Hijacking

In this version of the API, /attach, uses hijacking to transport stdin,
stdout and stderr on the same socket.

To hint potential proxies about connection hijacking, Docker client sends
connection upgrade headers similarly to websocket.

    Upgrade: tcp
    Connection: Upgrade

When Docker daemon detects the `Upgrade` header, it will switch its status code
from **200 OK** to **101 UPGRADED** and resend the same headers.

This might change in the future.

## 3.3 CORS Requests

To set cross origin requests to the remote api please give values to 
"--api-cors-header" when running docker in daemon mode. Set * will allow all,
default or blank means CORS disabled

    $ docker -d -H="192.168.1.9:2375" --api-cors-header="http://foo.bar"
