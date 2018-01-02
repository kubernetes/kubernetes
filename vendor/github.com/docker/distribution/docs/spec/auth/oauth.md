---
title: "Oauth2 Token Authentication"
description: "Specifies the Docker Registry v2 authentication"
keywords: ["registry, on-prem, images, tags, repository, distribution, oauth2, advanced"]
---

# Docker Registry v2 authentication using OAuth2

This document describes support for the OAuth2 protocol within the authorization
server. [RFC6749](https://tools.ietf.org/html/rfc6749) should be used as a
reference for the protocol and HTTP endpoints described here.

**Note**: Not all token servers implement oauth2. If the request to the endpoint
returns `404` using the HTTP `POST` method, refer to
[Token Documentation](token.md) for using the HTTP `GET` method supported by all
token servers.

## Refresh token format

The format of the refresh token is completely opaque to the client and should be
determined by the authorization server. The authorization should ensure the
token is sufficiently long and is responsible for storing any information about
long-lived tokens which may be needed for revoking. Any information stored
inside the token will not be extracted and presented by clients.

## Getting a token

POST /token

#### Headers
Content-Type: application/x-www-form-urlencoded

#### Post parameters

<dl>
    <dt>
        <code>grant_type</code>
    </dt>
    <dd>
        (REQUIRED) Type of grant used to get token. When getting a refresh token
        using credentials this type should be set to "password" and have the
        accompanying username and password paramters. Type "authorization_code"
        is reserved for future use for authenticating to an authorization server
        without having to send credentials directly from the client. When
        requesting an access token with a refresh token this should be set to
        "refresh_token".
    </dd>
    <dt>
        <code>service</code>
    </dt>
    <dd>
        (REQUIRED) The name of the service which hosts the resource to get
        access for. Refresh tokens will only be good for getting tokens for
        this service.
    </dd>
    <dt>
        <code>client_id</code>
    </dt>
    <dd>
        (REQUIRED) String identifying the client. This client_id does not need
        to be registered with the authorization server but should be set to a
        meaningful value in order to allow auditing keys created by unregistered
        clients. Accepted syntax is defined in
        [RFC6749 Appendix A.1](https://tools.ietf.org/html/rfc6749#appendix-A.1)
    </dd>
    <dt>
        <code>access_type</code>
    </dt>
    <dd>
        (OPTIONAL) Access which is being requested. If "offline" is provided
        then a refresh token will be returned. The default is "online" only
        returning short lived access token. If the grant type is "refresh_token"
        this will only return the same refresh token and not a new one.
    </dd>
    <dt>
        <code>scope</code>
    </dt>
    <dd>
        (OPTIONAL) The resource in question, formatted as one of the space-delimited
        entries from the <code>scope</code> parameters from the <code>WWW-Authenticate</code> header
        shown above. This query parameter should only be specified once but may
        contain multiple scopes using the scope list format defined in the scope
        grammar. If multiple <code>scope</code> is provided from
        <code>WWW-Authenticate</code> header the scopes should first be
        converted to a scope list before requesting the token. The above example
        would be specified as: <code>scope=repository:samalba/my-app:push</code>.
        When requesting a refresh token the scopes may be empty since the
        refresh token will not be limited by this scope, only the provided short
        lived access token will have the scope limitation.
    </dd>
    <dt>
        <code>refresh_token</code>
    </dt>
    <dd>
        (OPTIONAL) The refresh token to use for authentication when grant type "refresh_token" is used.
    </dd>
    <dt>
        <code>username</code>
    </dt>
    <dd>
        (OPTIONAL) The username to use for authentication when grant type "password" is used.
    </dd>
    <dt>
        <code>password</code>
    </dt>
    <dd>
        (OPTIONAL) The password to use for authentication when grant type "password" is used.
    </dd>
</dl>

#### Response fields

<dl>
    <dt>
        <code>access_token</code>
    </dt>
    <dd>
        (REQUIRED) An opaque <code>Bearer</code> token that clients should
        supply to subsequent requests in the <code>Authorization</code> header.
        This token should not be attempted to be parsed or understood by the
        client but treated as opaque string.
    </dd>
    <dt>
        <code>scope</code>
    </dt>
    <dd>
        (REQUIRED) The scope granted inside the access token. This may be the
        same scope as requested or a subset. This requirement is stronger than
        specified in [RFC6749 Section 4.2.2](https://tools.ietf.org/html/rfc6749#section-4.2.2)
        by strictly requiring the scope in the return value.
    </dd>
    <dt>
        <code>expires_in</code>
    </dt>
    <dd>
        (REQUIRED) The duration in seconds since the token was issued that it
        will remain valid.  When omitted, this defaults to 60 seconds.  For
        compatibility with older clients, a token should never be returned with
        less than 60 seconds to live.
    </dd>
    <dt>
        <code>issued_at</code>
    </dt>
    <dd>
        (Optional) The <a href="https://www.ietf.org/rfc/rfc3339.txt">RFC3339</a>-serialized UTC
        standard time at which a given token was issued. If <code>issued_at</code> is omitted, the
        expiration is from when the token exchange completed.
    </dd>
    <dt>
        <code>refresh_token</code>
    </dt>
    <dd>
        (Optional) Token which can be used to get additional access tokens for
        the same subject with different scopes. This token should be kept secure
        by the client and only sent to the authorization server which issues
        bearer tokens. This field will only be set when `access_type=offline` is
        provided in the request.
    </dd>
</dl>


#### Example getting refresh token

```
POST /token HTTP/1.1
Host: auth.docker.io
Content-Type: application/x-www-form-urlencoded

grant_type=password&username=johndoe&password=A3ddj3w&service=hub.docker.io&client_id=dockerengine&access_type=offline

HTTP/1.1 200 OK
Content-Type: application/json

{"refresh_token":"kas9Da81Dfa8","access_token":"eyJhbGciOiJFUzI1NiIsInR5","expires_in":900,"scope":""}
```

#### Example refreshing an Access Token

```
POST /token HTTP/1.1
Host: auth.docker.io
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&refresh_token=kas9Da81Dfa8&service=registry-1.docker.io&client_id=dockerengine&scope=repository:samalba/my-app:pull,push

HTTP/1.1 200 OK
Content-Type: application/json

{"refresh_token":"kas9Da81Dfa8","access_token":"eyJhbGciOiJFUzI1NiIsInR5":"expires_in":900,"scope":"repository:samalba/my-app:pull,repository:samalba/my-app:push"}
```
