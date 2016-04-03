<!--[metadata]>
+++
title = "Remote API"
description = "API Documentation for Docker"
keywords = ["API, Docker, rcli, REST,  documentation"]
[menu.main]
parent = "smn_remoteapi"
+++
<![end-metadata]-->

# Docker Remote API

 - By default the Docker daemon listens on `unix:///var/run/docker.sock`
   and the client must have `root` access to interact with the daemon.
 - If the Docker daemon is set to use an encrypted TCP socket (`--tls`,
   or `--tlsverify`) as with Boot2Docker 1.3.0, then you need to add extra
   parameters to `curl` or `wget` when making test API requests:
   `curl --insecure --cert ~/.docker/cert.pem --key ~/.docker/key.pem https://boot2docker:2376/images/json`
   or 
   `wget --no-check-certificate --certificate=$DOCKER_CERT_PATH/cert.pem --private-key=$DOCKER_CERT_PATH/key.pem https://boot2docker:2376/images/json -O - -q`
 - If a group named `docker` exists on your system, docker will apply
   ownership of the socket to the group.
 - The API tends to be REST, but for some complex commands, like attach
   or pull, the HTTP connection is hijacked to transport STDOUT, STDIN,
   and STDERR.
 - Since API version 1.2, the auth configuration is now handled client
   side, so the client has to send the `authConfig` as a `POST` in `/images/(name)/push`.
 - authConfig, set as the `X-Registry-Auth` header, is currently a Base64
   encoded (JSON) string with the following structure:
   `{"username": "string", "password": "string", "email": "string",
   "serveraddress" : "string", "auth": ""}`. Notice that `auth` is to be left
   empty, `serveraddress` is a domain/ip without protocol, and that double
   quotes (instead of single ones) are required.
 - The Remote API uses an open schema model.  In this model, unknown 
   properties in incoming messages will be ignored.
   Client applications need to take this into account to ensure
   they will not break when talking to newer Docker daemons.

The current version of the API is v1.20

Calling `/info` is the same as calling
`/v1.20/info`.

You can still call an old version of the API using
`/v1.19/info`.

## Docker Events

The following diagram depicts the container states accessible through the API.

![States](../images/event_state.png)

Some container-related events are not affected by container state, so they are not included in this diagram. These events are:

* **export** emitted by `docker export`
* **exec_create** emitted by `docker exec`
* **exec_start** emitted by `docker exec` after **exec_create**

Running `docker rmi` emits an **untag** event when removing an image name.  The `rmi` command may also emit **delete** events when images are deleted by ID directly or by deleting the last tag referring to the image.

> **Acknowledgement**: This diagram and the accompanying text were used with the permission of Matt Good and Gilder Labs. See Matt's original blog post [Docker Events Explained](http://gliderlabs.com/blog/2015/04/14/docker-events-explained/).

## v1.20

### Full documentation

[*Docker Remote API v1.20*](/reference/api/docker_remote_api_v1.20/)

### What's new

`GET /containers/(id)/archive`

**New!**
Get an archive of filesystem content from a container.

`PUT /containers/(id)/archive`

**New!**
Upload an archive of content to be extracted to an
existing directory inside a container's filesystem.

`POST /containers/(id)/copy`

**Deprecated!**
This copy endpoint has been deprecated in favor of the above `archive` endpoint
which can be used to download files and directories from a container.

**New!**
The `hostConfig` option now accepts the field `GroupAdd`, which specifies a list of additional
groups that the container process will run as.

## v1.19

### Full documentation

[*Docker Remote API v1.19*](/reference/api/docker_remote_api_v1.19/)

### What's new

**New!**
When the daemon detects a version mismatch with the client, usually when
the client is newer than the daemon, an HTTP 400 is now returned instead
of a 404.

`GET /containers/(id)/stats`

**New!**
You can now supply a `stream` bool to get only one set of stats and
disconnect

`GET /containers(id)/logs`

**New!**

This endpoint now accepts a `since` timestamp parameter.

`GET /info`

**New!**

The fields `Debug`, `IPv4Forwarding`, `MemoryLimit`, and `SwapLimit`
are now returned as boolean instead of as an int.

In addition, the end point now returns the new boolean fields
`CpuCfsPeriod`, `CpuCfsQuota`, and `OomKillDisable`.

## v1.18

### Full documentation

[*Docker Remote API v1.18*](/reference/api/docker_remote_api_v1.18/)

### What's new

`GET /version`

**New!**
This endpoint now returns `Os`, `Arch` and `KernelVersion`.

`POST /containers/create`
`POST /containers/(id)/start`

**New!**
You can set ulimit settings to be used within the container.

`GET /info`

**New!**
This endpoint now returns `SystemTime`, `HttpProxy`,`HttpsProxy` and `NoProxy`.

`GET /images/json`

**New!**
Added a `RepoDigests` field to include image digest information.

`POST /build`

**New!**
Builds can now set resource constraints for all containers created for the build.

**New!**
(`CgroupParent`) can be passed in the host config to setup container cgroups under a specific cgroup.

`POST /build`

**New!**
Closing the HTTP request will now cause the build to be canceled.

`POST /containers/(id)/exec`

**New!**
Add `Warnings` field to response.

## v1.17

### Full documentation

[*Docker Remote API v1.17*](/reference/api/docker_remote_api_v1.17/)

### What's new

The build supports `LABEL` command. Use this to add metadata
to an image. For example you could add data describing the content of an image.

`LABEL "com.example.vendor"="ACME Incorporated"`

**New!**
`POST /containers/(id)/attach` and `POST /exec/(id)/start`

**New!**
Docker client now hints potential proxies about connection hijacking using HTTP Upgrade headers.

`POST /containers/create`

**New!**
You can set labels on container create describing the container.

`GET /containers/json`

**New!**
The endpoint returns the labels associated with the containers (`Labels`).

`GET /containers/(id)/json`

**New!**
This endpoint now returns the list current execs associated with the container (`ExecIDs`).
This endpoint now returns the container labels (`Config.Labels`).

`POST /containers/(id)/rename`

**New!**
New endpoint to rename a container `id` to a new name.

`POST /containers/create`
`POST /containers/(id)/start`

**New!**
(`ReadonlyRootfs`) can be passed in the host config to mount the container's
root filesystem as read only.

`GET /containers/(id)/stats`

**New!**
This endpoint returns a live stream of a container's resource usage statistics.

`GET /images/json`

**New!**
This endpoint now returns the labels associated with each image (`Labels`).


## v1.16

### Full documentation

[*Docker Remote API v1.16*](/reference/api/docker_remote_api_v1.16/)

### What's new

`GET /info`

**New!**
`info` now returns the number of CPUs available on the machine (`NCPU`),
total memory available (`MemTotal`), a user-friendly name describing the running Docker daemon (`Name`), a unique ID identifying the daemon (`ID`), and
a list of daemon labels (`Labels`).

`POST /containers/create`

**New!**
You can set the new container's MAC address explicitly.

**New!**
Volumes are now initialized when the container is created.

`POST /containers/(id)/copy`

**New!**
You can now copy data which is contained in a volume.

## v1.15

### Full documentation

[*Docker Remote API v1.15*](/reference/api/docker_remote_api_v1.15/)

### What's new

`POST /containers/create`

**New!**
It is now possible to set a container's HostConfig when creating a container.
Previously this was only available when starting a container.

## v1.14

### Full documentation

[*Docker Remote API v1.14*](/reference/api/docker_remote_api_v1.14/)

### What's new

`DELETE /containers/(id)`

**New!**
When using `force`, the container will be immediately killed with SIGKILL.

`POST /containers/(id)/start`

**New!**
The `hostConfig` option now accepts the field `CapAdd`, which specifies a list of capabilities
to add, and the field `CapDrop`, which specifies a list of capabilities to drop.

`POST /images/create`

**New!**
The `fromImage` and `repo` parameters now supports the `repo:tag` format.
Consequently,  the `tag` parameter is now obsolete. Using the new format and
the `tag` parameter at the same time will return an error.

## v1.13

### Full documentation

[*Docker Remote API v1.13*](/reference/api/docker_remote_api_v1.13/)

### What's new

`GET /containers/(name)/json`

**New!**
The `HostConfig.Links` field is now filled correctly

**New!**
`Sockets` parameter added to the `/info` endpoint listing all the sockets the 
daemon is configured to listen on.

`POST /containers/(name)/start`
`POST /containers/(name)/stop`

**New!**
`start` and `stop` will now return 304 if the container's status is not modified

`POST /commit`

**New!**
Added a `pause` parameter (default `true`) to pause the container during commit

## v1.12

### Full documentation

[*Docker Remote API v1.12*](/reference/api/docker_remote_api_v1.12/)

### What's new

`POST /build`

**New!**
Build now has support for the `forcerm` parameter to always remove containers

`GET /containers/(name)/json`
`GET /images/(name)/json`

**New!**
All the JSON keys are now in CamelCase

**New!**
Trusted builds are now Automated Builds - `is_trusted` is now `is_automated`.

**Removed Insert Endpoint**
The `insert` endpoint has been removed.

## v1.11

### Full documentation

[*Docker Remote API v1.11*](/reference/api/docker_remote_api_v1.11/)

### What's new

`GET /_ping`

**New!**
You can now ping the server via the `_ping` endpoint.

`GET /events`

**New!**
You can now use the `-until` parameter to close connection
after timestamp.

`GET /containers/(id)/logs`

This url is preferred method for getting container logs now.

## v1.10

### Full documentation

[*Docker Remote API v1.10*](/reference/api/docker_remote_api_v1.10/)

### What's new

`DELETE /images/(name)`

**New!**
You can now use the force parameter to force delete of an
    image, even if it's tagged in multiple repositories. **New!**
    You
    can now use the noprune parameter to prevent the deletion of parent
    images

`DELETE /containers/(id)`

**New!**
You can now use the force parameter to force delete a
    container, even if it is currently running

## v1.9

### Full documentation

[*Docker Remote API v1.9*](/reference/api/docker_remote_api_v1.9/)

### What's new

`POST /build`

**New!**
This endpoint now takes a serialized ConfigFile which it
uses to resolve the proper registry auth credentials for pulling the
base image. Clients which previously implemented the version
accepting an AuthConfig object must be updated.

## v1.8

### Full documentation

[*Docker Remote API v1.8*](/reference/api/docker_remote_api_v1.8/)

### What's new

`POST /build`

**New!**
This endpoint now returns build status as json stream. In
case of a build error, it returns the exit status of the failed
command.

`GET /containers/(id)/json`

**New!**
This endpoint now returns the host config for the
container.

`POST /images/create`

`POST /images/(name)/insert`

`POST /images/(name)/push`

**New!**
progressDetail object was added in the JSON. It's now
possible to get the current value and the total of the progress
without having to parse the string.

## v1.7

### Full documentation

[*Docker Remote API v1.7*](/reference/api/docker_remote_api_v1.7/)

### What's new

`GET /images/json`

The format of the json returned from this uri changed. Instead of an
entry for each repo/tag on an image, each image is only represented
once, with a nested attribute indicating the repo/tags that apply to
that image.

Instead of:

    HTTP/1.1 200 OK
    Content-Type: application/json

    [
      {
        "VirtualSize": 131506275,
        "Size": 131506275,
        "Created": 1365714795,
        "Id": "8dbd9e392a964056420e5d58ca5cc376ef18e2de93b5cc90e868a1bbc8318c1c",
        "Tag": "12.04",
        "Repository": "ubuntu"
      },
      {
        "VirtualSize": 131506275,
        "Size": 131506275,
        "Created": 1365714795,
        "Id": "8dbd9e392a964056420e5d58ca5cc376ef18e2de93b5cc90e868a1bbc8318c1c",
        "Tag": "latest",
        "Repository": "ubuntu"
      },
      {
        "VirtualSize": 131506275,
        "Size": 131506275,
        "Created": 1365714795,
        "Id": "8dbd9e392a964056420e5d58ca5cc376ef18e2de93b5cc90e868a1bbc8318c1c",
        "Tag": "precise",
        "Repository": "ubuntu"
      },
      {
        "VirtualSize": 180116135,
        "Size": 24653,
        "Created": 1364102658,
        "Id": "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
        "Tag": "12.10",
        "Repository": "ubuntu"
      },
      {
        "VirtualSize": 180116135,
        "Size": 24653,
        "Created": 1364102658,
        "Id": "b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc",
        "Tag": "quantal",
        "Repository": "ubuntu"
      }
    ]

The returned json looks like this:

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

`GET /images/viz`

This URI no longer exists. The `images --viz`
output is now generated in the client, using the
`/images/json` data.

## v1.6

### Full documentation

[*Docker Remote API v1.6*](/reference/api/docker_remote_api_v1.6/)

### What's new

`POST /containers/(id)/attach`

**New!**
You can now split stderr from stdout. This is done by
prefixing a header to each transmission. See
[`POST /containers/(id)/attach`](
/reference/api/docker_remote_api_v1.9/#attach-to-a-container "POST /containers/(id)/attach").
The WebSocket attach is unchanged. Note that attach calls on the
previous API version didn't change. Stdout and stderr are merged.

## v1.5

### Full documentation

[*Docker Remote API v1.5*](/reference/api/docker_remote_api_v1.5/)

### What's new

`POST /images/create`

**New!**
You can now pass registry credentials (via an AuthConfig
    object) through the X-Registry-Auth header

`POST /images/(name)/push`

**New!**
The AuthConfig object now needs to be passed through the
    X-Registry-Auth header

`GET /containers/json`

**New!**
The format of the Ports entry has been changed to a list of
dicts each containing PublicPort, PrivatePort and Type describing a
port mapping.

## v1.4

### Full documentation

[*Docker Remote API v1.4*](/reference/api/docker_remote_api_v1.4/)

### What's new

`POST /images/create`

**New!**
When pulling a repo, all images are now downloaded in parallel.

`GET /containers/(id)/top`

**New!**
You can now use ps args with docker top, like docker top
    <container_id> aux

`GET /events`

**New!**
Image's name added in the events

## v1.3

docker v0.5.0
[51f6c4a](https://github.com/docker/docker/commit/51f6c4a7372450d164c61e0054daf0223ddbd909)

### Full documentation

[*Docker Remote API v1.3*](/reference/api/docker_remote_api_v1.3/)

### What's new

`GET /containers/(id)/top`

List the processes running inside a container.

`GET /events`

**New!**
Monitor docker's events via streaming or via polling

Builder (/build):

 - Simplify the upload of the build context
 - Simply stream a tarball instead of multipart upload with 4
   intermediary buffers
 - Simpler, less memory usage, less disk usage and faster

> **Warning**: 
> The /build improvements are not reverse-compatible. Pre 1.3 clients will
> break on /build.

List containers (/containers/json):

 - You can use size=1 to get the size of the containers

Start containers (/containers/<id>/start):

 - You can now pass host-specific configuration (e.g., bind mounts) in
   the POST body for start calls

## v1.2

docker v0.4.2
[2e7649b](https://github.com/docker/docker/commit/2e7649beda7c820793bd46766cbc2cfeace7b168)

### Full documentation

[*Docker Remote API v1.2*](/reference/api/docker_remote_api_v1.2/)

### What's new

The auth configuration is now handled by the client.

The client should send it's authConfig as POST on each call of
`/images/(name)/push`

`GET /auth`

**Deprecated.**

`POST /auth`

Only checks the configuration but doesn't store it on the server

    Deleting an image is now improved, will only untag the image if it
    has children and remove all the untagged parents if has any.

`POST /images/<name>/delete`

Now returns a JSON structure with the list of images
deleted/untagged.

## v1.1

docker v0.4.0
[a8ae398](https://github.com/docker/docker/commit/a8ae398bf52e97148ee7bd0d5868de2e15bd297f)

### Full documentation

[*Docker Remote API v1.1*](/reference/api/docker_remote_api_v1.1/)

### What's new

`POST /images/create`

`POST /images/(name)/insert`

`POST /images/(name)/push`

Uses json stream instead of HTML hijack, it looks like this:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {"status":"Pushing..."}
        {"status":"Pushing", "progress":"1/? (n/a)"}
        {"error":"Invalid..."}
        ...

## v1.0

docker v0.3.4
[8d73740](https://github.com/docker/docker/commit/8d73740343778651c09160cde9661f5f387b36f4)

### Full documentation

[*Docker Remote API v1.0*](/reference/api/docker_remote_api_v1.0/)

### What's new

Initial version
