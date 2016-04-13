<!--[metadata]>
+++
draft = true
title = "Remote API v1.4"
description = "API Documentation for Docker"
keywords = ["API, Docker, rcli, REST,  documentation"]
[menu.main]
parent="smn_remoteapi"
+++
<![end-metadata]-->

# Docker Remote API v1.4

# 1. Brief introduction

- The Remote API is replacing rcli
- Default port in the docker daemon is 2375
- The API tends to be REST, but for some complex commands, like attach
  or pull, the HTTP connection is hijacked to transport stdout stdin
  and stderr

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
                     "Image": "ubuntu:latest",
                     "Command": "echo 1",
                     "Created": 1367854155,
                     "Status": "Exit 0",
                     "Ports":"",
                     "SizeRw":12288,
                     "SizeRootFs":0
             },
             {
                     "Id": "9cd87474be90",
                     "Image": "ubuntu:latest",
                     "Command": "echo 222222",
                     "Created": 1367854155,
                     "Status": "Exit 0",
                     "Ports":"",
                     "SizeRw":12288,
                     "SizeRootFs":0
             },
             {
                     "Id": "3176a2479c92",
                     "Image": "centos:latest",
                     "Command": "echo 3333333333333333",
                     "Created": 1367854154,
                     "Status": "Exit 0",
                     "Ports":"",
                     "SizeRw":12288,
                     "SizeRootFs":0
             },
             {
                     "Id": "4cb07b47f9fb",
                     "Image": "fedora:latest",
                     "Command": "echo 444444444444444444444444444444444",
                     "Created": 1367854152,
                     "Status": "Exit 0",
                     "Ports":"",
                     "SizeRw":12288,
                     "SizeRootFs":0
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
             "AttachStdin":false,
             "AttachStdout":true,
             "AttachStderr":true,
             "PortSpecs":null,
             "Privileged": false,
             "Tty":false,
             "OpenStdin":false,
             "StdinOnce":false,
             "Env":null,
             "Cmd":[
                     "date"
             ],
             "Dns":null,
             "Image":"ubuntu",
             "Volumes":{},
             "VolumesFrom":"",
             "WorkingDir":""

        }

**Example response**:

        HTTP/1.1 201 Created
        Content-Type: application/json

        {
             "Id":"e90e34656806"
             "Warnings":[]
        }

Json Parameters:

-   **config** – the container's configuration

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
                             "Image": "ubuntu",
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
                     "Volumes": {}
        }

Status Codes:

-   **200** – no error
-   **404** – no such container
-   **409** – conflict between containers and images
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
             "LxcConf":[{"Key":"lxc.utsname","Value":"docker"}]
        }

**Example response**:

        HTTP/1.1 204 No Content
        Content-Type: text/plain

Json Parameters:

     

-   **hostConfig** – the container's host configuration (optional)

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

### List Images

`GET /images/(format)`

List images `format` could be json or viz (json default)

**Example request**:

        GET /images/json?all=0 HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
             {
                     "Repository":"ubuntu",
                     "Tag":"precise",
                     "Id":"b750fe79269d",
                     "Created":1364102658,
                     "Size":24653,
                     "VirtualSize":180116135
             },
             {
                     "Repository":"ubuntu",
                     "Tag":"12.04",
                     "Id":"b750fe79269d",
                     "Created":1364102658,
                     "Size":24653,
                     "VirtualSize":180116135
             }
        ]

**Example request**:

        GET /images/viz HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: text/plain

        digraph docker {
        "d82cbacda43a" -> "074be284591f"
        "1496068ca813" -> "08306dc45919"
        "08306dc45919" -> "0e7893146ac2"
        "b750fe79269d" -> "1496068ca813"
        base -> "27cf78414709" [style=invis]
        "f71189fff3de" -> "9a33b36209ed"
        "27cf78414709" -> "b750fe79269d"
        "0e7893146ac2" -> "d6434d954665"
        "d6434d954665" -> "d82cbacda43a"
        base -> "e9aa60c60128" [style=invis]
        "074be284591f" -> "f71189fff3de"
        "b750fe79269d" [label="b750fe79269d\nubuntu",shape=box,fillcolor="paleturquoise",style="filled,rounded"];
        "e9aa60c60128" [label="e9aa60c60128\ncentos",shape=box,fillcolor="paleturquoise",style="filled,rounded"];
        "9a33b36209ed" [label="9a33b36209ed\nfedora",shape=box,fillcolor="paleturquoise",style="filled,rounded"];
        base [style=invisible]
        }

Query Parameters:

-   **all** – 1/True/true or 0/False/false, Show all containers.
        Only running containers are shown by defaul

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Create an image

`POST /images/create`

Create an image, either by pull it from the registry or by importing i

**Example request**:

        POST /images/create?fromImage=ubuntu HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"status":"Pulling..."}
        {"status":"Pulling", "progress":"1/? (n/a)"}
        {"error":"Invalid..."}
        ...

Query Parameters:

-   **fromImage** – name of the image to pull
-   **fromSrc** – source to import, - means stdin
-   **repo** – repository
-   **tag** – tag
-   **registry** – the registry to pull from

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
        {"status":"Inserting", "progress":"1/? (n/a)"}
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

        GET /images/centos/json HTTP/1.1

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
                             "Image":"centos",
                             "Volumes":null,
                             "VolumesFrom":"",
                             "WorkingDir":""
                     },
             "Size": 6824592
        }

Status Codes:

-   **200** – no error
-   **404** – no such image
-   **409** – conflict between containers and images
-   **500** – server error

### Get the history of an image

`GET /images/(name)/history`

Return the history of the image `name`

**Example request**:

        GET /images/fedora/history HTTP/1.1

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
        {{ authConfig }}

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

    {"status":"Pushing..."} {"status":"Pushing", "progress":"1/? (n/a)"}
    {"error":"Invalid..."} ...

Status Codes:

-   **200** – no error :statuscode 404: no such image :statuscode
        500: server error

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

Status Codes:

-   **200** – no error
-   **404** – no such image
-   **409** – conflict
-   **500** – server error

### Search images

`GET /images/search`

Search for an image on [Docker Hub](https://hub.docker.com)

**Example request**:

        GET /images/search?term=sshd HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
             {
                     "Name":"cespare/sshd",
                     "Description":""
             },
             {
                     "Name":"johnfuller/sshd",
                     "Description":""
             },
             {
                     "Name":"dhrp/mongodb-sshd",
                     "Description":""
             }
        ]

        :query term: term to search
        :statuscode 200: no error
        :statuscode 500: server error

## 2.3 Misc

### Build an image from Dockerfile via stdin

`POST /build`

Build an image from Dockerfile via stdin

**Example request**:

        POST /build HTTP/1.1

        {{ TAR STREAM }}

**Example response**:

        HTTP/1.1 200 OK

        {{ STREAM }}

    The stream must be a tar archive compressed with one of the
    following algorithms: identity (no compression), gzip, bzip2, xz.
    The archive must include a file called Dockerfile at its root. I
    may include any number of other files, which will be accessible in
    the build context (See the ADD build command).

    The Content-type header should be set to "application/tar".

Query Parameters:

-   **t** – repository name (and optionally a tag) to be applied to
    the resulting image in case of success
-   **remote** – build source URI (git or HTTPS/HTTP)
-   **q** – suppress verbose build output
-   **nocache** – do not use the cache when building the image

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

### Show the docker version information

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
            "Cmd": ["cat", "/world"],
            "PortSpecs":["22"]
        }

**Example response**:

        HTTP/1.1 201 OK
            Content-Type: application/vnd.docker.raw-stream

        {"Id": "596069db4bf5"}

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

        {"status":"create","id":"dfdf82bd3881","from":"ubuntu:latest","time":1374067924}
        {"status":"start","id":"dfdf82bd3881","from":"ubuntu:latest","time":1374067924}
        {"status":"stop","id":"dfdf82bd3881","from":"ubuntu:latest","time":1374067966}
        {"status":"destroy","id":"dfdf82bd3881","from":"ubuntu:latest","time":1374067970}

Query Parameters:

-   **since** – timestamp used for polling

Status Codes:

-   **200** – no error
-   **500** – server error

# 3. Going further

## 3.1 Inside `docker run`

Here are the steps of `docker run` :

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
stdout and stderr on the same socket. This might change in the future.

## 3.3 CORS Requests

To enable cross origin requests to the remote api add the flag
"--api-enable-cors" when running docker in daemon mode.

    $ docker -d -H="192.168.1.9:2375" --api-enable-cors
