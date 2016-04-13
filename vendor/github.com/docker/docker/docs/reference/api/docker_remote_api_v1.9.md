<!--[metadata]>
+++
draft = true
title = "Remote API v1.9"
description = "API Documentation for Docker"
keywords = ["API, Docker, rcli, REST,  documentation"]
[menu.main]
parent="smn_remoteapi"
+++
<![end-metadata]-->

# Docker Remote API v1.9

# 1. Brief introduction

 - The Remote API has replaced rcli
 - The daemon listens on `unix:///var/run/docker.sock` but you can bind
   Docker to another host/port or a Unix socket.
 - The API tends to be REST, but for some complex commands, like `attach`
   or `pull`, the HTTP connection is hijacked to transport `stdout, stdin`
   and `stderr`

# 2. Endpoints

## 2.1 Containers

### List containers

`GET /containers/json`

List containers.

**Example request**:

        GET /containers/json?all=1&before=8dfafdbc3a40&size=1 HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
             {
                     "Id": "8dfafdbc3a40",
                     "Image": "base:latest",
                     "Command": "echo 1",
                     "Created": 1367854155,
                     "Status": "Exit 0",
                     "Ports": [{"PrivatePort": 2222, "PublicPort": 3333, "Type": "tcp"}],
                     "SizeRw": 12288,
                     "SizeRootFs": 0
             },
             {
                     "Id": "9cd87474be90",
                     "Image": "base:latest",
                     "Command": "echo 222222",
                     "Created": 1367854155,
                     "Status": "Exit 0",
                     "Ports": [],
                     "SizeRw": 12288,
                     "SizeRootFs": 0
             },
             {
                     "Id": "3176a2479c92",
                     "Image": "base:latest",
                     "Command": "echo 3333333333333333",
                     "Created": 1367854154,
                     "Status": "Exit 0",
                     "Ports":[],
                     "SizeRw":12288,
                     "SizeRootFs":0
             },
             {
                     "Id": "4cb07b47f9fb",
                     "Image": "base:latest",
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
-   **limit** – Show `limit` last created containers, include non-running ones.
-   **since** – Show only containers created since Id, include non-running ones.
-   **before** – Show only containers created before Id, include non-running ones.
-   **size** – 1/True/true or 0/False/false, Show the containers sizes

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
             "Hostname":"",
             "User":"",
             "Memory":0,
             "MemorySwap":0,
             "CpuShares":0,
             "AttachStdin":false,
             "AttachStdout":true,
             "AttachStderr":true,
             "PortSpecs":null,
             "Tty":false,
             "OpenStdin":false,
             "StdinOnce":false,
             "Env":null,
             "Cmd":[
                     "date"
             ],
             "Dns":null,
             "Image":"base",
             "Volumes":{
                     "/tmp": {}
             },
             "VolumesFrom":"",
             "WorkingDir":"",
             "ExposedPorts":{
                     "22/tcp": {}
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

     

-   **Hostname** – Container host name
-   **User** – Username or UID
-   **Memory** – Memory Limit in bytes
-   **CpuShares** – CPU shares (relative weight)
-   **AttachStdin** – 1/True/true or 0/False/false, attach to
        standard input. Default false
-   **AttachStdout** – 1/True/true or 0/False/false, attach to
        standard output. Default false
-   **AttachStderr** – 1/True/true or 0/False/false, attach to
        standard error. Default false
-   **Tty** – 1/True/true or 0/False/false, allocate a pseudo-tty.
        Default false
-   **OpenStdin** – 1/True/true or 0/False/false, keep stdin open
        even if not attached. Default false

Query Parameters:

     

-   **name** – Assign the specified name to the container. Mus
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
                     "Id": "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2",
                     "Created": "2013-05-07T14:51:42.041847+02:00",
                     "Path": "date",
                     "Args": [],
                     "Config": {
                             "Hostname": "4fa6e0f0c678",
                             "User": "",
                             "Memory": 0,
                             "MemorySwap": 0,
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
                             "Dns": null,
                             "Image": "base",
                             "Volumes": {},
                             "VolumesFrom": "",
                             "WorkingDir": ""
                     },
                     "State": {
                             "Running": false,
                             "Pid": 0,
                             "ExitCode": 0,
                             "StartedAt": "2013-05-07T14:51:42.087658+02:01360",
                             "Ghost": false
                     },
                     "Image": "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
                     "NetworkSettings": {
                             "IpAddress": "",
                             "IpPrefixLen": 0,
                             "Gateway": "",
                             "Bridge": "",
                             "PortMapping": null
                     },
                     "SysInitPath": "/home/kitty/go/src/github.com/docker/docker/bin/docker",
                     "ResolvConfPath": "/etc/resolv.conf",
                     "Volumes": {},
                     "HostConfig": {
                         "Binds": null,
                         "ContainerIDFile": "",
                         "LxcConf": [],
                         "Privileged": false,
                         "PortBindings": {
                            "80/tcp": [
                                {
                                    "HostIp": "0.0.0.0",
                                    "HostPort": "49153"
                                }
                            ]
                         },
                         "Links": null,
                         "PublishAllPorts": false
                     }
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

### Start a container

`POST /containers/(id)/start`

Start the container `id`

**Example request**:

        POST /containers/(id)/start HTTP/1.1
        Content-Type: application/json

        {
             "Binds":["/tmp:/tmp"],
             "LxcConf":[{"Key":"lxc.utsname","Value":"docker"}],
             "PortBindings":{ "22/tcp": [{ "HostPort": "11022" }] },
             "PublishAllPorts":false,
             "Privileged":false
        }

**Example response**:

        HTTP/1.1 204 No Content
        Content-Type: text/plain

Json Parameters:

     

-   **Binds** – Create a bind mount to a directory or file with
        [host-path]:[container-path]:[rw|ro]. If a directory
        "container-path" is missing, then docker creates a new volume.
-   **LxcConf** – Map of custom lxc options
-   **PortBindings** – Expose ports from the container, optionally
        publishing them via the HostPort flag
-   **PublishAllPorts** – 1/True/true or 0/False/false, publish all
        exposed ports to the host interfaces. Default false
-   **Privileged** – 1/True/true or 0/False/false, give extended
        privileges to this container. Default false

Status Codes:

-   **204** – no error
-   **404** – no such container
-   **500** – server error

### Stop a container

`POST /containers/(id)/stop`

Stop the container `id`

**Example request**:

        POST /containers/e90e34656806/stop?t=5 HTTP/1.1

**Example response**:

        HTTP/1.1 204 OK

Query Parameters:

-   **t** – number of seconds to wait before killing the container

Status Codes:

-   **204** – no error
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
    When not set, SIGKILL is assumed and the call will wait for the container to exit.

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

        HTTP/1.1 200 OK
        Content-Type: application/vnd.docker.raw-stream

        {{ STREAM }}

Query Parameters:

-   **logs** – 1/True/true or 0/False/false, return logs. Defaul
        false
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

    **Stream details**:

    When using the TTY setting is enabled in
    [`POST /containers/create`](#create-a-container), the
    stream is the raw data from the process PTY and client's stdin. When
    the TTY is disabled, then the stream is multiplexed to separate
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
    5.  Goto 1)

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
        Content-Type: application/octet-stream

        {{ TAR STREAM }}

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **500** – server error

## 2.2 Images

### List images

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

### Create an image

`POST /images/create`

Create an image, either by pull it from the registry or by importing i

**Example request**:

        POST /images/create?fromImage=base HTTP/1.1

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
-   **fromSrc** – source to import, - means stdin
-   **repo** – repository
-   **tag** – tag
-   **registry** – the registry to pull from

Request Headers:

-   **X-Registry-Auth** – base64-encoded AuthConfig object

Status Codes:

-   **200** – no error
-   **500** – server error

### Insert a file in an image

`POST /images/(name)/insert`

Insert a file from `url` in the image `name` at `path`

**Example request**:

        POST /images/test/insert?path=/usr&url=myurl HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"status":"Inserting..."}
        {"status":"Inserting", "progress":"1/? (n/a)", "progressDetail":{"current":1}}
        {"error":"Invalid..."}
        ...

Query Parameters:

-	**url** – The url from where the file is taken
-	**path** – The path where the file is stored

Status Codes:

-   **200** – no error
-   **500** – server error

### Inspect an image

`GET /images/(name)/json`

Return low-level information on the image `name`

**Example request**:

        GET /images/base/json HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
             "id":"b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
             "parent":"27cf784147099545",
             "created":"2013-03-23T22:24:18.818426-07:00",
             "container":"3d67245a8d72ecf13f33dffac9f79dcdf70f75acb84d308770391510e0c23ad0",
             "container_config":
                     {
                             "Hostname":"",
                             "User":"",
                             "Memory":0,
                             "MemorySwap":0,
                             "AttachStdin":false,
                             "AttachStdout":false,
                             "AttachStderr":false,
                             "PortSpecs":null,
                             "Tty":true,
                             "OpenStdin":true,
                             "StdinOnce":false,
                             "Env":null,
                             "Cmd": ["/bin/bash"],
                             "Dns":null,
                             "Image":"base",
                             "Volumes":null,
                             "VolumesFrom":"",
                             "WorkingDir":""
                     },
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

        GET /images/base/history HTTP/1.1

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

`DELETE /images/(name*)
:   Remove the image `name` from the filesystem

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
                    "is_trusted": false,
                    "name": "wma55/u1210sshd",
                    "star_count": 0
                },
                {
                    "description": "",
                    "is_official": false,
                    "is_trusted": false,
                    "name": "jdswinbank/sshd",
                    "star_count": 0
                },
                {
                    "description": "",
                    "is_official": false,
                    "is_trusted": false,
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

### Build an image from Dockerfile

`POST /build`

Build an image from Dockerfile using a POST body.

**Example request**:

        POST /build HTTP/1.1

        {{ TAR STREAM }}

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"stream": "Step 1..."}
        {"stream": "..."}
        {"error": "Error...", "errorDetail": {"code": 123, "message": "Error..."}}

    The stream must be a tar archive compressed with one of the
    following algorithms: identity (no compression), gzip, bzip2, xz.

    The archive must include a file called `Dockerfile`
 at its root. It may include any number of other files,
    which will be accessible in the build context (See the [*ADD build
    command*](/reference/builder/#add)).

Query Parameters:

-   **t** – repository name (and optionally a tag) to be applied to
    the resulting image in case of success
-   **remote** – build source URI (git or HTTPS/HTTP)
-   **q** – suppress verbose build output
-   **nocache** – do not use the cache when building the image
-   **rm** – Remove intermediate containers after a successful build

    Request Headers:

-   **Content-type** – should be set to `"application/tar"`.
-   **X-Registry-Config** – base64-encoded ConfigFile object

Status Codes:

-   **200** – no error
-   **500** – server error

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
        Content-Type: text/plain

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
             "Containers":11,
             "Images":16,
             "Debug":false,
             "NFd": 11,
             "NGoroutines":21,
             "MemoryLimit":true,
             "SwapLimit":false,
             "IPv4Forwarding":true
        }

Status Codes:

-   **200** – no error
-   **500** – server error

### Show the Docker version information

`GET /version`

Show the docker version information

**Example request**:

        GET /version HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
             "Version":"0.2.2",
             "GitCommit":"5a2a5cc+CHANGES",
             "GoVersion":"go1.0.3"
        }

Status Codes:

-   **200** – no error
-   **500** – server error

### Create a new image from a container's changes

`POST /commit`

Create a new image from a container's changes

**Example request**:

        POST /commit?container=44c004db4b17&m=message&repo=myrepo HTTP/1.1
        Content-Type: application/json

        {
             "Hostname":"",
             "User":"",
             "Memory":0,
             "MemorySwap":0,
             "AttachStdin":false,
             "AttachStdout":true,
             "AttachStderr":true,
             "PortSpecs":null,
             "Tty":false,
             "OpenStdin":false,
             "StdinOnce":false,
             "Env":null,
             "Cmd":[
                     "date"
             ],
             "Volumes":{
                     "/tmp": {}
             },
             "WorkingDir":"",
             "DisableNetwork": false,
             "ExposedPorts":{
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
-   **m** – commit message
-   **author** – author (e.g., "John Hannibal Smith
    <[hannibal@a-team.com](mailto:hannibal%40a-team.com)>")

Status Codes:

-   **201** – no error
-   **404** – no such container
-   **500** – server error

### Monitor Docker's events

`GET /events`

Get events from docker, either in real time via streaming, or via
polling (using since).

Docker containers will report the following events:

    create, destroy, die, export, kill, pause, restart, start, stop, unpause

and Docker images will report:

    untag, delete

**Example request**:

        GET /events?since=1374067924

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"status": "create", "id": "dfdf82bd3881","from": "base:latest", "time":1374067924}
        {"status": "start", "id": "dfdf82bd3881","from": "base:latest", "time":1374067924}
        {"status": "stop", "id": "dfdf82bd3881","from": "base:latest", "time":1374067966}
        {"status": "destroy", "id": "dfdf82bd3881","from": "base:latest", "time":1374067970}

Query Parameters:

-   **since** – timestamp used for polling

Status Codes:

-   **200** – no error
-   **500** – server error

### Get a tarball containing all images and tags in a repository

`GET /images/(name)/get`

Get a tarball containing all images and metadata for the repository specified by `name`.

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

# 3. Going further

## 3.1 Inside `docker run`

Here are the steps of `docker run` :

 - Create the container

 - If the status code is 404, it means the image doesn't exist:

- Try to pull it
-   Then retry to create the container

 - Start the container

 - If you are not in detached mode:

- Attach to the container, using logs=1 (to have stdout and
- stderr from the container's start) and stream=1

 - If in detached mode or only stdin is attached:

- Display the container's id

## 3.2 Hijacking

In this version of the API, /attach, uses hijacking to transport stdin,
stdout and stderr on the same socket. This might change in the future.

## 3.3 CORS requests

To enable cross origin requests to the remote api add the flag
"--api-enable-cors" when running docker in daemon mode.

    $ docker -d -H="192.168.1.9:2375" --api-enable-cors
