<!--[metadata]>
+++
title = "Docker Hub API"
description = "API Documentation for the Docker Hub API"
keywords = ["API, Docker, index, REST, documentation, Docker Hub,  registry"]
[menu.main]
parent = "smn_remoteapi"
+++
<![end-metadata]-->

# Docker Hub API

- This is the REST API for [Docker Hub](https://hub.docker.com).
- Authorization is done with basic auth over SSL
- Not all commands require authentication, only those noted as such.

# Repositories

## User repository

### Create a user repository

`PUT /v1/repositories/(namespace)/(repo_name)/`

Create a user repository with the given `namespace` and `repo_name`.

**Example Request**:

        PUT /v1/repositories/foo/bar/ HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Basic akmklmasadalkm==
        X-Docker-Token: true

        [{"id": "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f"}]

Parameters:

- **namespace** – the namespace for the repo
- **repo_name** – the name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        WWW-Authenticate: Token signature=123abc,repository="foo/bar",access=write
        X-Docker-Token: signature=123abc,repository="foo/bar",access=write
        X-Docker-Endpoints: registry-1.docker.io [, registry-2.docker.io]

        ""

Status Codes:

- **200** – Created
- **400** – Errors (invalid json, missing or invalid fields, etc)
- **401** – Unauthorized
- **403** – Account is not Active

### Delete a user repository

`DELETE /v1/repositories/(namespace)/(repo_name)/`

Delete a user repository with the given `namespace` and `repo_name`.

**Example Request**:

        DELETE /v1/repositories/foo/bar/ HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Basic akmklmasadalkm==
        X-Docker-Token: true

        ""

Parameters:

- **namespace** – the namespace for the repo
- **repo_name** – the name for the repo

**Example Response**:

        HTTP/1.1 202
        Vary: Accept
        Content-Type: application/json
        WWW-Authenticate: Token signature=123abc,repository="foo/bar",access=delete
        X-Docker-Token: signature=123abc,repository="foo/bar",access=delete
        X-Docker-Endpoints: registry-1.docker.io [, registry-2.docker.io]

        ""

Status Codes:

- **200** – Deleted
- **202** – Accepted
- **400** – Errors (invalid json, missing or invalid fields, etc)
- **401** – Unauthorized
- **403** – Account is not Active

## Library repository

### Create a library repository

`PUT /v1/repositories/(repo_name)/`

Create a library repository with the given `repo_name`.
This is a restricted feature only available to docker admins.

> When namespace is missing, it is assumed to be `library`


**Example Request**:

        PUT /v1/repositories/foobar/ HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Basic akmklmasadalkm==
        X-Docker-Token: true

        [{"id": "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f"}]

Parameters:

- **repo_name** – the library name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json
        WWW-Authenticate: Token signature=123abc,repository="library/foobar",access=write
        X-Docker-Token: signature=123abc,repository="foo/bar",access=write
        X-Docker-Endpoints: registry-1.docker.io [, registry-2.docker.io]

        ""

Status Codes:

- **200** – Created
- **400** – Errors (invalid json, missing or invalid fields, etc)
- **401** – Unauthorized
- **403** – Account is not Active

### Delete a library repository

`DELETE /v1/repositories/(repo_name)/`

Delete a library repository with the given `repo_name`.
This is a restricted feature only available to docker admins.

> When namespace is missing, it is assumed to be `library`


**Example Request**:

        DELETE /v1/repositories/foobar/ HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Basic akmklmasadalkm==
        X-Docker-Token: true

        ""

Parameters:

- **repo_name** – the library name for the repo

**Example Response**:

        HTTP/1.1 202
        Vary: Accept
        Content-Type: application/json
        WWW-Authenticate: Token signature=123abc,repository="library/foobar",access=delete
        X-Docker-Token: signature=123abc,repository="foo/bar",access=delete
        X-Docker-Endpoints: registry-1.docker.io [, registry-2.docker.io]

        ""

Status Codes:

- **200** – Deleted
- **202** – Accepted
- **400** – Errors (invalid json, missing or invalid fields, etc)
- **401** – Unauthorized
- **403** – Account is not Active

# Repository images

## User repository images

### Update user repository images

`PUT /v1/repositories/(namespace)/(repo_name)/images`

Update the images for a user repo.

**Example Request**:

        PUT /v1/repositories/foo/bar/images HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Basic akmklmasadalkm==

        [{"id": "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f",
        "checksum": "b486531f9a779a0c17e3ed29dae8f12c4f9e89cc6f0bc3c38722009fe6857087"}]

Parameters:

- **namespace** – the namespace for the repo
- **repo_name** – the name for the repo

**Example Response**:

        HTTP/1.1 204
        Vary: Accept
        Content-Type: application/json

        ""

Status Codes:

- **204** – Created
- **400** – Errors (invalid json, missing or invalid fields, etc)
- **401** – Unauthorized
- **403** – Account is not Active or permission denied

### List user repository images

`GET /v1/repositories/(namespace)/(repo_name)/images`

Get the images for a user repo.

**Example Request**:

        GET /v1/repositories/foo/bar/images HTTP/1.1
        Host: index.docker.io
        Accept: application/json

Parameters:

- **namespace** – the namespace for the repo
- **repo_name** – the name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json

        [{"id": "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f",
        "checksum": "b486531f9a779a0c17e3ed29dae8f12c4f9e89cc6f0bc3c38722009fe6857087"},
        {"id": "ertwetewtwe38722009fe6857087b486531f9a779a0c1dfddgfgsdgdsgds",
        "checksum": "34t23f23fc17e3ed29dae8f12c4f9e89cc6f0bsdfgfsdgdsgdsgerwgew"}]

Status Codes:

- **200** – OK
- **404** – Not found

## Library repository images

### Update library repository images

`PUT /v1/repositories/(repo_name)/images`

Update the images for a library repo.

**Example Request**:

        PUT /v1/repositories/foobar/images HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Basic akmklmasadalkm==

        [{"id": "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f",
        "checksum": "b486531f9a779a0c17e3ed29dae8f12c4f9e89cc6f0bc3c38722009fe6857087"}]

Parameters:

- **repo_name** – the library name for the repo

**Example Response**:

        HTTP/1.1 204
        Vary: Accept
        Content-Type: application/json

        ""

Status Codes:

- **204** – Created
- **400** – Errors (invalid json, missing or invalid fields, etc)
- **401** – Unauthorized
- **403** – Account is not Active or permission denied

### List library repository images

`GET /v1/repositories/(repo_name)/images`

Get the images for a library repo.

**Example Request**:

        GET /v1/repositories/foobar/images HTTP/1.1
        Host: index.docker.io
        Accept: application/json

Parameters:

- **repo_name** – the library name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json

        [{"id": "9e89cc6f0bc3c38722009fe6857087b486531f9a779a0c17e3ed29dae8f12c4f",
        "checksum": "b486531f9a779a0c17e3ed29dae8f12c4f9e89cc6f0bc3c38722009fe6857087"},
        {"id": "ertwetewtwe38722009fe6857087b486531f9a779a0c1dfddgfgsdgdsgds",
        "checksum": "34t23f23fc17e3ed29dae8f12c4f9e89cc6f0bsdfgfsdgdsgdsgerwgew"}]

Status Codes:

- **200** – OK
- **404** – Not found

# Repository authorization

## Library repository

### Authorize a token for a library

`PUT /v1/repositories/(repo_name)/auth`

Authorize a token for a library repo

**Example Request**:

        PUT /v1/repositories/foobar/auth HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Authorization: Token signature=123abc,repository="library/foobar",access=write

Parameters:

- **repo_name** – the library name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json

        "OK"

Status Codes:

- **200** – OK
- **403** – Permission denied
- **404** – Not found

## User repository

### Authorize a token for a user repository

`PUT /v1/repositories/(namespace)/(repo_name)/auth`

Authorize a token for a user repo

**Example Request**:

        PUT /v1/repositories/foo/bar/auth HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Authorization: Token signature=123abc,repository="foo/bar",access=write

Parameters:

- **namespace** – the namespace for the repo
- **repo_name** – the name for the repo

**Example Response**:

        HTTP/1.1 200
        Vary: Accept
        Content-Type: application/json

        "OK"

Status Codes:

- **200** – OK
- **403** – Permission denied
- **404** – Not found

## Users

### User login

`GET /v1/users/`

If you want to check your login, you can try this endpoint

**Example Request**:

        GET /v1/users/ HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Authorization: Basic akmklmasadalkm==

**Example Response**:

        HTTP/1.1 200 OK
        Vary: Accept
        Content-Type: application/json

        OK

Status Codes:

- **200** – no error
- **401** – Unauthorized
- **403** – Account is not Active

### User register

`POST /v1/users/`

Registering a new account.

**Example request**:

        POST /v1/users/ HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json

        {"email": "sam@docker.com",
         "password": "toto42",
         "username": "foobar"}

Json Parameters:

- **email** – valid email address, that needs to be confirmed
- **username** – min 4 character, max 30 characters, must match
        the regular expression [a-z0-9_].
- **password** – min 5 characters

**Example Response**:

        HTTP/1.1 201 OK
        Vary: Accept
        Content-Type: application/json

        "User Created"

Status Codes:

- **201** – User Created
- **400** – Errors (invalid json, missing or invalid fields, etc)

### Update user

`PUT /v1/users/(username)/`

Change a password or email address for given user. If you pass in an
email, it will add it to your account, it will not remove the old
one. Passwords will be updated.

It is up to the client to verify that that password that is sent is
the one that they want. Common approach is to have them type it
twice.

**Example Request**:

        PUT /v1/users/fakeuser/ HTTP/1.1
        Host: index.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Basic akmklmasadalkm==

        {"email": "sam@docker.com",
         "password": "toto42"}

Parameters:

- **username** – username for the person you want to update

**Example Response**:

        HTTP/1.1 204
        Vary: Accept
        Content-Type: application/json

        ""

Status Codes:

- **204** – User Updated
- **400** – Errors (invalid json, missing or invalid fields, etc)
- **401** – Unauthorized
- **403** – Account is not Active
- **404** – User not found
