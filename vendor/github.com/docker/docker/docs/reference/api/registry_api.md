<!--[metadata]>
+++
draft = true
title = "Registry v1 API"
description = "API Documentation for Docker Registry"
keywords = ["API, Docker, index, registry, REST,  documentation"]
[menu.main]
parent="smn_registry_ref"
+++
<![end-metadata]-->

# Docker Registry API v1

## Introduction

 - This is the REST API for the Docker Registry 1.0
 - It stores the images and the graph for a set of repositories
 - It does not have user accounts data
 - It has no notion of user accounts or authorization
 - It delegates authentication and authorization to the Index Auth
   service using tokens
 - It supports different storage backends (S3, cloud files, local FS)
 - It doesn't have a local database
 - The registry is open source: [Docker Registry](https://github.com/docker/docker-registry)

 We expect that there will be multiple registries out there. To help to
grasp the context, here are some examples of registries:

 - **sponsor registry**: such a registry is provided by a third-party
   hosting infrastructure as a convenience for their customers and the
   Docker community as a whole. Its costs are supported by the third
   party, but the management and operation of the registry are
   supported by Docker. It features read/write access, and delegates
   authentication and authorization to the Index.
 - **mirror registry**: such a registry is provided by a third-party
   hosting infrastructure but is targeted at their customers only. Some
   mechanism (unspecified to date) ensures that public images are
   pulled from a sponsor registry to the mirror registry, to make sure
   that the customers of the third-party provider can `docker pull`
   those images locally.
 - **vendor registry**: such a registry is provided by a software
   vendor, who wants to distribute Docker images. It would be operated
   and managed by the vendor. Only users authorized by the vendor would
   be able to get write access. Some images would be public (accessible
   for anyone), others private (accessible only for authorized users).
   Authentication and authorization would be delegated to the Index.
   The goal of vendor registries is to let someone do `docker pull
   basho/riak1.3` and automatically push from the vendor registry
   (instead of a sponsor registry); i.e., get all the convenience of a
   sponsor registry, while retaining control on the asset distribution.
 - **private registry**: such a registry is located behind a firewall,
   or protected by an additional security layer (HTTP authorization,
   SSL client-side certificates, IP address authorization...). The
   registry is operated by a private entity, outside of Docker's
   control. It can optionally delegate additional authorization to the
   Index, but it is not mandatory.

> **Note**:
> Mirror registries and private registries which do not use the Index
> don't even need to run the registry code. They can be implemented by any
> kind of transport implementing HTTP GET and PUT. Read-only registries
> can be powered by a simple static HTTPS server.

> **Note**:
> The latter implies that while HTTP is the protocol of choice for a registry,
> multiple schemes are possible (and in some cases, trivial):
>
>  - HTTP with GET (and PUT for read-write registries);
>  - local mount point;
>  - remote Docker addressed through SSH.

The latter would only require two new commands in Docker, e.g.,
`registryget` and `registryput`, wrapping access to the local filesystem
(and optionally doing consistency checks). Authentication and authorization
are then delegated to SSH (e.g., with public keys).

> **Note**:
> Private registry servers that expose an HTTP endpoint need to be secured with
> TLS (preferably TLSv1.2, but at least TLSv1.0). Make sure to put the CA
> certificate at /etc/docker/certs.d/my.registry.com:5000/ca.crt on the Docker
> host, so that the daemon can securely access the private registry.
> Support for SSLv3 and lower is not available due to security issues.

The default namespace for a private repository is `library`.

# Endpoints

## Images

### Get image layer

`GET /v1/images/(image_id)/layer`

Get image layer for a given `image_id`

**Example Request**:

        GET /v1/images/088b4505aa3adc3d35e79c031fa126b403200f02f51920fbd9b7c503e87c7a2c/layer HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Token signature=123abc,repository="foo/bar",access=read

Parameters:

- **image_id** – the id for the layer you want to get

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        X-Docker-Registry-Version: 0.6.0
        Cookie: (Cookie provided by the Registry)

        {layer binary data stream}

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Image not found

### Put image layer

`PUT /v1/images/(image_id)/layer`

Put image layer for a given `image_id`

**Example Request**:

        PUT /v1/images/088b4505aa3adc3d35e79c031fa126b403200f02f51920fbd9b7c503e87c7a2c/layer HTTP/1.1
        Host: registry-1.docker.io
        Transfer-Encoding: chunked
        Authorization: Token signature=123abc,repository="foo/bar",access=write

        {layer binary data stream}

Parameters:

- **image_id** – the id for the layer you want to get

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        ""

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Image not found

## Image

### Put image layer

`PUT /v1/images/(image_id)/json`

Put image for a given `image_id`

**Example Request**:

        PUT /v1/images/088b4505aa3adc3d35e79c031fa126b403200f02f51920fbd9b7c503e87c7a2c/json HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        Cookie: (Cookie provided by the Registry)

        {
            id: "088b4505aa3adc3d35e79c031fa126b403200f02f51920fbd9b7c503e87c7a2c",
            parent: "aeee6396d62273d180a49c96c62e45438d87c7da4a5cf5d2be6bee4e21bc226f",
            created: "2013-04-30T17:46:10.843673+03:00",
            container: "8305672a76cc5e3d168f97221106ced35a76ec7ddbb03209b0f0d96bf74f6ef7",
            container_config: {
                Hostname: "host-test",
                User: "",
                Memory: 0,
                MemorySwap: 0,
                AttachStdin: false,
                AttachStdout: false,
                AttachStderr: false,
                Tty: false,
                OpenStdin: false,
                StdinOnce: false,
                Env: null,
                Cmd: [
                "/bin/bash",
                "-c",
                "apt-get -q -yy -f install libevent-dev"
                ],
                Dns: null,
                Image: "imagename/blah",
                Volumes: { },
                VolumesFrom: ""
            },
            docker_version: "0.1.7"
        }

Parameters:

- **image_id** – the id for the layer you want to get

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        ""

Status Codes:

- **200** – OK
- **401** – Requires authorization

### Get image layer

`GET /v1/images/(image_id)/json`

Get image for a given `image_id`

**Example Request**:

        GET /v1/images/088b4505aa3adc3d35e79c031fa126b403200f02f51920fbd9b7c503e87c7a2c/json HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        Cookie: (Cookie provided by the Registry)

Parameters:

- **image_id** – the id for the layer you want to get

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0
        X-Docker-Size: 456789
        X-Docker-Checksum: b486531f9a779a0c17e3ed29dae8f12c4f9e89cc6f0bc3c38722009fe6857087

        {
            id: "088b4505aa3adc3d35e79c031fa126b403200f02f51920fbd9b7c503e87c7a2c",
            parent: "aeee6396d62273d180a49c96c62e45438d87c7da4a5cf5d2be6bee4e21bc226f",
            created: "2013-04-30T17:46:10.843673+03:00",
            container: "8305672a76cc5e3d168f97221106ced35a76ec7ddbb03209b0f0d96bf74f6ef7",
            container_config: {
                Hostname: "host-test",
                User: "",
                Memory: 0,
                MemorySwap: 0,
                AttachStdin: false,
                AttachStdout: false,
                AttachStderr: false,
                Tty: false,
                OpenStdin: false,
                StdinOnce: false,
                Env: null,
                Cmd: [
                "/bin/bash",
                "-c",
                "apt-get -q -yy -f install libevent-dev"
                ],
                Dns: null,
                Image: "imagename/blah",
                Volumes: { },
                VolumesFrom: ""
            },
            docker_version: "0.1.7"
        }

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Image not found

## Ancestry

### Get image ancestry

`GET /v1/images/(image_id)/ancestry`

Get ancestry for an image given an `image_id`

**Example Request**:

        GET /v1/images/088b4505aa3adc3d35e79c031fa126b403200f02f51920fbd9b7c503e87c7a2c/ancestry HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        Cookie: (Cookie provided by the Registry)

Parameters:

- **image_id** – the id for the layer you want to get

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        ["088b4502f51920fbd9b7c503e87c7a2c05aa3adc3d35e79c031fa126b403200f",
         "aeee63968d87c7da4a5cf5d2be6bee4e21bc226fd62273d180a49c96c62e4543",
         "bfa4c5326bc764280b0863b46a4b20d940bc1897ef9c1dfec060604bdc383280",
         "6ab5893c6927c15a15665191f2c6cf751f5056d8b95ceee32e43c5e8a3648544"]

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Image not found

## Tags

### List repository tags

`GET /v1/repositories/(namespace)/(repository)/tags`

Get all of the tags for the given repo.

**Example Request**:

        GET /v1/repositories/reynholm/help-system-server/tags HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0
        Cookie: (Cookie provided by the Registry)

Parameters:

- **namespace** – namespace for the repo
- **repository** – name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        {
            "latest": "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f",
            "0.1.1":  "b486531f9a779a0c17e3ed29dae8f12c4f9e89cc6f0bc3c38722009fe6857087"
        }

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Repository not found

### Get image id for a particular tag

`GET /v1/repositories/(namespace)/(repository)/tags/(tag*)`

Get a tag for the given repo.

**Example Request**:

        GET /v1/repositories/reynholm/help-system-server/tags/latest HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0
        Cookie: (Cookie provided by the Registry)

Parameters:

- **namespace** – namespace for the repo
- **repository** – name for the repo
- **tag** – name of tag you want to get

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f"

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Tag not found

### Delete a repository tag

`DELETE /v1/repositories/(namespace)/(repository)/tags/(tag*)`

Delete the tag for the repo

**Example Request**:

        DELETE /v1/repositories/reynholm/help-system-server/tags/latest HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        Cookie: (Cookie provided by the Registry)

Parameters:

- **namespace** – namespace for the repo
- **repository** – name for the repo
- **tag** – name of tag you want to delete

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        ""

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Tag not found

### Set a tag for a specified image id

`PUT /v1/repositories/(namespace)/(repository)/tags/(tag*)`

Put a tag for the given repo.

**Example Request**:

        PUT /v1/repositories/reynholm/help-system-server/tags/latest HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        Cookie: (Cookie provided by the Registry)

        "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f"

Parameters:

- **namespace** – namespace for the repo
- **repository** – name for the repo
- **tag** – name of tag you want to add

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        ""

Status Codes:

- **200** – OK
- **400** – Invalid data
- **401** – Requires authorization
- **404** – Image not found

## Repositories

### Delete a repository

`DELETE /v1/repositories/(namespace)/(repository)/`

Delete a repository

**Example Request**:

        DELETE /v1/repositories/reynholm/help-system-server/ HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json
        Cookie: (Cookie provided by the Registry)

        ""

Parameters:

- **namespace** – namespace for the repo
- **repository** – name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        ""

Status Codes:

- **200** – OK
- **401** – Requires authorization
- **404** – Repository not found

## Search

If you need to search the index, this is the endpoint you would use.

`GET /v1/search`

Search the Index given a search term. It accepts

    [GET](http://www.w3.org/Protocols/rfc2616/rfc2616-sec9.html#sec9.3)
    only.

**Example request**:

        GET /v1/search?q=search_term&page=1&n=25 HTTP/1.1
        Host: index.docker.io
        Accept: application/json

Query Parameters:

- **q** – what you want to search for
- **n** - number of results you want returned per page (default: 25, min:1, max:100)
- **page** - page number of results

**Example response**:

        HTTP/1.1 200 OK
        Vary: Accept
        Content-Type: application/json

        {"num_pages": 1,
          "num_results": 3,
          "results" : [
             {"name": "ubuntu", "description": "An ubuntu image..."},
             {"name": "centos", "description": "A centos image..."},
             {"name": "fedora", "description": "A fedora image..."}
           ],
          "page_size": 25,
          "query":"search_term",
          "page": 1
         }

Response Items:
- **num_pages** - Total number of pages returned by query
- **num_results** - Total number of results returned by query
- **results** - List of results for the current page
- **page_size** - How many results returned per page
- **query** - Your search term
- **page** - Current page number

Status Codes:

- **200** – no error
- **500** – server error

## Status

### Status check for registry

`GET /v1/_ping`

Check status of the registry. This endpoint is also used to
determine if the registry supports SSL.

**Example Request**:

        GET /v1/_ping HTTP/1.1
        Host: registry-1.docker.io
        Accept: application/json
        Content-Type: application/json

        ""

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        X-Docker-Registry-Version: 0.6.0

        ""

Status Codes:

- **200** – OK

## Authorization

This is where we describe the authorization process, including the
tokens and cookies.

