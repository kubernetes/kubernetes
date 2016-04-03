<!--[metadata]>
+++
title = "docker.io accounts API"
description = "API Documentation for docker.io accounts."
keywords = ["API, Docker, accounts, REST,  documentation"]
[menu.main]
parent = "mn_reference"
+++
<![end-metadata]-->

# docker.io accounts API

## Get a single user

`GET /api/v1.1/users/:username/`

Get profile info for the specified user.

Parameters:

-   **username** – username of the user whose profile info is being
        requested.

Request Headers:

-   **Authorization** – required authentication credentials of
        either type HTTP Basic or OAuth Bearer Token.

Status Codes:

-   **200** – success, user data returned.
-   **401** – authentication error.
-   **403** – permission error, authenticated user must be the user
        whose data is being requested, OAuth access tokens must have
        `profile_read` scope.
-   **404** – the specified username does not exist.

**Example request**:

        GET /api/v1.1/users/janedoe/ HTTP/1.1
        Host: www.docker.io
        Accept: application/json
        Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
            "id": 2,
            "username": "janedoe",
            "url": "https://www.docker.io/api/v1.1/users/janedoe/",
            "date_joined": "2014-02-12T17:58:01.431312Z",
            "type": "User",
            "full_name": "Jane Doe",
            "location": "San Francisco, CA",
            "company": "Success, Inc.",
            "profile_url": "https://docker.io/",
            "gravatar_url": "https://secure.gravatar.com/avatar/0212b397124be4acd4e7dea9aa357.jpg?s=80&r=g&d=mm"
            "email": "jane.doe@example.com",
            "is_active": true
        }

## Update a single user

`PATCH /api/v1.1/users/:username/`

Update profile info for the specified user.

Parameters:

-   **username** – username of the user whose profile info is being
        updated.

Json Parameters:

-   **full_name** (*string*) – (optional) the new name of the user.
-   **location** (*string*) – (optional) the new location.
-   **company** (*string*) – (optional) the new company of the user.
-   **profile_url** (*string*) – (optional) the new profile url.
-   **gravatar_email** (*string*) – (optional) the new Gravatar
        email address.

Request Headers:

-   **Authorization** – required authentication credentials of
        either type HTTP Basic or OAuth Bearer Token.
-   **Content-Type** – MIME Type of post data. JSON, url-encoded
        form data, etc.

Status Codes:

-   **200** – success, user data updated.
-   **400** – post data validation error.
-   **401** – authentication error.
-   **403** – permission error, authenticated user must be the user
        whose data is being updated, OAuth access tokens must have
        `profile_write` scope.
-   **404** – the specified username does not exist.

**Example request**:

        PATCH /api/v1.1/users/janedoe/ HTTP/1.1
        Host: www.docker.io
        Accept: application/json
        Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=

        {
            "location": "Private Island",
            "profile_url": "http://janedoe.com/",
            "company": "Retired",
        }

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
            "id": 2,
            "username": "janedoe",
            "url": "https://www.docker.io/api/v1.1/users/janedoe/",
            "date_joined": "2014-02-12T17:58:01.431312Z",
            "type": "User",
            "full_name": "Jane Doe",
            "location": "Private Island",
            "company": "Retired",
            "profile_url": "http://janedoe.com/",
            "gravatar_url": "https://secure.gravatar.com/avatar/0212b397124be4acd4e7dea9aa357.jpg?s=80&r=g&d=mm"
            "email": "jane.doe@example.com",
            "is_active": true
        }

## List email addresses for a user

`GET /api/v1.1/users/:username/emails/`

List email info for the specified user.

Parameters:

-   **username** – username of the user whose profile info is being
        updated.

Request Headers:

-   **Authorization** – required authentication credentials of
        either type HTTP Basic or OAuth Bearer Token

Status Codes:

-   **200** – success, user data updated.
-   **401** – authentication error.
-   **403** – permission error, authenticated user must be the user
        whose data is being requested, OAuth access tokens must have
        `email_read` scope.
-   **404** – the specified username does not exist.

**Example request**:

        GET /api/v1.1/users/janedoe/emails/ HTTP/1.1
        Host: www.docker.io
        Accept: application/json
        Authorization: Bearer zAy0BxC1wDv2EuF3tGs4HrI6qJp6KoL7nM

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
            {
                "email": "jane.doe@example.com",
                "verified": true,
                "primary": true
            }
        ]

## Add email address for a user

`POST /api/v1.1/users/:username/emails/`

Add a new email address to the specified user's account. The email
address must be verified separately, a confirmation email is not
automatically sent.

Json Parameters:

-   **email** (*string*) – email address to be added.

Request Headers:

-   **Authorization** – required authentication credentials of
        either type HTTP Basic or OAuth Bearer Token.
-   **Content-Type** – MIME Type of post data. JSON, url-encoded
        form data, etc.

Status Codes:

-   **201** – success, new email added.
-   **400** – data validation error.
-   **401** – authentication error.
-   **403** – permission error, authenticated user must be the user
        whose data is being requested, OAuth access tokens must have
        `email_write` scope.
-   **404** – the specified username does not exist.

**Example request**:

        POST /api/v1.1/users/janedoe/emails/ HTTP/1.1
        Host: www.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Bearer zAy0BxC1wDv2EuF3tGs4HrI6qJp6KoL7nM

        {
            "email": "jane.doe+other@example.com"
        }

**Example response**:

        HTTP/1.1 201 Created
        Content-Type: application/json

        {
            "email": "jane.doe+other@example.com",
            "verified": false,
            "primary": false
        }

## Delete email address for a user

`DELETE /api/v1.1/users/:username/emails/`

Delete an email address from the specified user's account. You
cannot delete a user's primary email address.

Json Parameters:

-   **email** (*string*) – email address to be deleted.

Request Headers:

-   **Authorization** – required authentication credentials of
        either type HTTP Basic or OAuth Bearer Token.
-   **Content-Type** – MIME Type of post data. JSON, url-encoded
        form data, etc.

Status Codes:

-   **204** – success, email address removed.
-   **400** – validation error.
-   **401** – authentication error.
-   **403** – permission error, authenticated user must be the user
        whose data is being requested, OAuth access tokens must have
        `email_write` scope.
-   **404** – the specified username or email address does not
        exist.

**Example request**:

        DELETE /api/v1.1/users/janedoe/emails/ HTTP/1.1
        Host: www.docker.io
        Accept: application/json
        Content-Type: application/json
        Authorization: Bearer zAy0BxC1wDv2EuF3tGs4HrI6qJp6KoL7nM

        {
            "email": "jane.doe+other@example.com"
        }

**Example response**:

        HTTP/1.1 204 NO CONTENT
        Content-Length: 0
