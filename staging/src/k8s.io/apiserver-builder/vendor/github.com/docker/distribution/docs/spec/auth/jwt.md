<!--[metadata]>
+++
title = "Token Authentication Implementation"
description = "Describe the reference implementation of the Docker Registry v2 authentication schema"
keywords = ["registry, on-prem, images, tags, repository, distribution, JWT authentication, advanced"]
[menu.main]
parent="smn_registry_ref"
+++
<![end-metadata]-->

# Docker Registry v2 Bearer token specification

This specification covers the `docker/distribution` implementation of the
v2 Registry's authentication schema.  Specifically, it describes the JSON
Web Token schema that `docker/distribution` has adopted to implement the
client-opaque Bearer token issued by an authentication service and
understood by the registry.

This document borrows heavily from the [JSON Web Token Draft Spec](https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32)

## Getting a Bearer Token

For this example, the client makes an HTTP GET request to the following URL:

```
https://auth.docker.io/token?service=registry.docker.io&scope=repository:samalba/my-app:pull,push
```

The token server should first attempt to authenticate the client using any
authentication credentials provided with the request. As of Docker 1.8, the
registry client in the Docker Engine only supports Basic Authentication to
these token servers. If an attempt to authenticate to the token server fails,
the token server should return a `401 Unauthorized` response indicating that
the provided credentials are invalid.

Whether the token server requires authentication is up to the policy of that
access control provider. Some requests may require authentication to determine
access (such as pushing or pulling a private repository) while others may not
(such as pulling from a public repository).

After authenticating the client (which may simply be an anonymous client if
no attempt was made to authenticate), the token server must next query its
access control list to determine whether the client has the requested scope. In
this example request, if I have authenticated as user `jlhawn`, the token
server will determine what access I have to the repository `samalba/my-app`
hosted by the entity `registry.docker.io`.

Once the token server has determined what access the client has to the
resources requested in the `scope` parameter, it will take the intersection of
the set of requested actions on each resource and the set of actions that the
client has in fact been granted. If the client only has a subset of the
requested access **it must not be considered an error** as it is not the
responsibility of the token server to indicate authorization errors as part of
this workflow.

Continuing with the example request, the token server will find that the
client's set of granted access to the repository is `[pull, push]` which when
intersected with the requested access `[pull, push]` yields an equal set. If
the granted access set was found only to be `[pull]` then the intersected set
would only be `[pull]`. If the client has no access to the repository then the
intersected set would be empty, `[]`.

It is this intersected set of access which is placed in the returned token.

The server will now construct a JSON Web Token to sign and return. A JSON Web
Token has 3 main parts:

1.  Headers

    The header of a JSON Web Token is a standard JOSE header. The "typ" field
    will be "JWT" and it will also contain the "alg" which identifies the
    signing algorithm used to produce the signature. It will also usually have
    a "kid" field, the ID of the key which was used to sign the token.

    Here is an example JOSE Header for a JSON Web Token (formatted with
    whitespace for readability):

    ```
    {
        "typ": "JWT",
        "alg": "ES256",
        "kid": "PYYO:TEWU:V7JH:26JV:AQTZ:LJC3:SXVJ:XGHA:34F2:2LAQ:ZRMK:Z7Q6"
    }
    ```

    It specifies that this object is going to be a JSON Web token signed using
    the key with the given ID using the Elliptic Curve signature algorithm
    using a SHA256 hash.

2.  Claim Set

    The Claim Set is a JSON struct containing these standard registered claim
    name fields:

    <dl>
        <dt>
            <code>iss</code> (Issuer)
        </dt>
        <dd>
            The issuer of the token, typically the fqdn of the authorization
            server.
        </dd>
        <dt>
            <code>sub</code> (Subject)
        </dt>
        <dd>
            The subject of the token; the name or id of the client which
            requested it. This should be empty (`""`) if the client did not
            authenticate.
        </dd>
        <dt>
            <code>aud</code> (Audience)
        </dt>
        <dd>
            The intended audience of the token; the name or id of the service
            which will verify the token to authorize the client/subject.
        </dd>
        <dt>
            <code>exp</code> (Expiration)
        </dt>
        <dd>
            The token should only be considered valid up to this specified date
            and time.
        </dd>
        <dt>
            <code>nbf</code> (Not Before)
        </dt>
        <dd>
            The token should not be considered valid before this specified date
            and time.
        </dd>
        <dt>
            <code>iat</code> (Issued At)
        </dt>
        <dd>
            Specifies the date and time which the Authorization server
            generated this token.
        </dd>
        <dt>
            <code>jti</code> (JWT ID)
        </dt>
        <dd>
            A unique identifier for this token. Can be used by the intended
            audience to prevent replays of the token.
        </dd>
    </dl>

    The Claim Set will also contain a private claim name unique to this
    authorization server specification:

    <dl>
        <dt>
            <code>access</code>
        </dt>
        <dd>
            An array of access entry objects with the following fields:

            <dl>
                <dt>
                    <code>type</code>
                </dt>
                <dd>
                    The type of resource hosted by the service.
                </dd>
                <dt>
                    <code>name</code>
                </dt>
                <dd>
                    The name of the resource of the given type hosted by the
                    service.
                </dd>
                <dt>
                    <code>actions</code>
                </dt>
                <dd>
                    An array of strings which give the actions authorized on
                    this resource.
                </dd>
            </dl>
        </dd>
    </dl>

    Here is an example of such a JWT Claim Set (formatted with whitespace for
    readability):

    ```
    {
        "iss": "auth.docker.com",
        "sub": "jlhawn",
        "aud": "registry.docker.com",
        "exp": 1415387315,
        "nbf": 1415387015,
        "iat": 1415387015,
        "jti": "tYJCO1c6cnyy7kAn0c7rKPgbV1H1bFws",
        "access": [
            {
                "type": "repository",
                "name": "samalba/my-app",
                "actions": [
                    "pull",
                    "push"
                ]
            }
        ]
    }
    ```

3.  Signature

    The authorization server will produce a JOSE header and Claim Set with no
    extraneous whitespace, i.e., the JOSE Header from above would be

    ```
    {"typ":"JWT","alg":"ES256","kid":"PYYO:TEWU:V7JH:26JV:AQTZ:LJC3:SXVJ:XGHA:34F2:2LAQ:ZRMK:Z7Q6"}
    ```

    and the Claim Set from above would be

    ```
    {"iss":"auth.docker.com","sub":"jlhawn","aud":"registry.docker.com","exp":1415387315,"nbf":1415387015,"iat":1415387015,"jti":"tYJCO1c6cnyy7kAn0c7rKPgbV1H1bFws","access":[{"type":"repository","name":"samalba/my-app","actions":["push","pull"]}]}
    ```

    The utf-8 representation of this JOSE header and Claim Set are then
    url-safe base64 encoded (sans trailing '=' buffer), producing:

    ```
    eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6IlBZWU86VEVXVTpWN0pIOjI2SlY6QVFUWjpMSkMzOlNYVko6WEdIQTozNEYyOjJMQVE6WlJNSzpaN1E2In0
    ```

    for the JOSE Header and

    ```
    eyJpc3MiOiJhdXRoLmRvY2tlci5jb20iLCJzdWIiOiJqbGhhd24iLCJhdWQiOiJyZWdpc3RyeS5kb2NrZXIuY29tIiwiZXhwIjoxNDE1Mzg3MzE1LCJuYmYiOjE0MTUzODcwMTUsImlhdCI6MTQxNTM4NzAxNSwianRpIjoidFlKQ08xYzZjbnl5N2tBbjBjN3JLUGdiVjFIMWJGd3MiLCJhY2Nlc3MiOlt7InR5cGUiOiJyZXBvc2l0b3J5IiwibmFtZSI6InNhbWFsYmEvbXktYXBwIiwiYWN0aW9ucyI6WyJwdXNoIl19XX0
    ```

    for the Claim Set. These two are concatenated using a '.' character,
    yielding the string:

    ```
    eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6IlBZWU86VEVXVTpWN0pIOjI2SlY6QVFUWjpMSkMzOlNYVko6WEdIQTozNEYyOjJMQVE6WlJNSzpaN1E2In0.eyJpc3MiOiJhdXRoLmRvY2tlci5jb20iLCJzdWIiOiJqbGhhd24iLCJhdWQiOiJyZWdpc3RyeS5kb2NrZXIuY29tIiwiZXhwIjoxNDE1Mzg3MzE1LCJuYmYiOjE0MTUzODcwMTUsImlhdCI6MTQxNTM4NzAxNSwianRpIjoidFlKQ08xYzZjbnl5N2tBbjBjN3JLUGdiVjFIMWJGd3MiLCJhY2Nlc3MiOlt7InR5cGUiOiJyZXBvc2l0b3J5IiwibmFtZSI6InNhbWFsYmEvbXktYXBwIiwiYWN0aW9ucyI6WyJwdXNoIl19XX0
    ```

    This is then used as the payload to a the `ES256` signature algorithm
    specified in the JOSE header and specified fully in [Section 3.4 of the JSON Web Algorithms (JWA)
    draft specification](https://tools.ietf.org/html/draft-ietf-jose-json-web-algorithms-38#section-3.4)

    This example signature will use the following ECDSA key for the server:

    ```
    {
        "kty": "EC",
        "crv": "P-256",
        "kid": "PYYO:TEWU:V7JH:26JV:AQTZ:LJC3:SXVJ:XGHA:34F2:2LAQ:ZRMK:Z7Q6",
        "d": "R7OnbfMaD5J2jl7GeE8ESo7CnHSBm_1N2k9IXYFrKJA",
        "x": "m7zUpx3b-zmVE5cymSs64POG9QcyEpJaYCD82-549_Q",
        "y": "dU3biz8sZ_8GPB-odm8Wxz3lNDr1xcAQQPQaOcr1fmc"
    }
    ```

    A resulting signature of the above payload using this key is:

    ```
    QhflHPfbd6eVF4lM9bwYpFZIV0PfikbyXuLx959ykRTBpe3CYnzs6YBK8FToVb5R47920PVLrh8zuLzdCr9t3w
    ```

    Concatenating all of these together with a `.` character gives the
    resulting JWT:

    ```
    eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6IlBZWU86VEVXVTpWN0pIOjI2SlY6QVFUWjpMSkMzOlNYVko6WEdIQTozNEYyOjJMQVE6WlJNSzpaN1E2In0.eyJpc3MiOiJhdXRoLmRvY2tlci5jb20iLCJzdWIiOiJqbGhhd24iLCJhdWQiOiJyZWdpc3RyeS5kb2NrZXIuY29tIiwiZXhwIjoxNDE1Mzg3MzE1LCJuYmYiOjE0MTUzODcwMTUsImlhdCI6MTQxNTM4NzAxNSwianRpIjoidFlKQ08xYzZjbnl5N2tBbjBjN3JLUGdiVjFIMWJGd3MiLCJhY2Nlc3MiOlt7InR5cGUiOiJyZXBvc2l0b3J5IiwibmFtZSI6InNhbWFsYmEvbXktYXBwIiwiYWN0aW9ucyI6WyJwdXNoIl19XX0.QhflHPfbd6eVF4lM9bwYpFZIV0PfikbyXuLx959ykRTBpe3CYnzs6YBK8FToVb5R47920PVLrh8zuLzdCr9t3w
    ```

This can now be placed in an HTTP response and returned to the client to use to
authenticate to the audience service:


```
HTTP/1.1 200 OK
Content-Type: application/json

{"token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6IlBZWU86VEVXVTpWN0pIOjI2SlY6QVFUWjpMSkMzOlNYVko6WEdIQTozNEYyOjJMQVE6WlJNSzpaN1E2In0.eyJpc3MiOiJhdXRoLmRvY2tlci5jb20iLCJzdWIiOiJqbGhhd24iLCJhdWQiOiJyZWdpc3RyeS5kb2NrZXIuY29tIiwiZXhwIjoxNDE1Mzg3MzE1LCJuYmYiOjE0MTUzODcwMTUsImlhdCI6MTQxNTM4NzAxNSwianRpIjoidFlKQ08xYzZjbnl5N2tBbjBjN3JLUGdiVjFIMWJGd3MiLCJhY2Nlc3MiOlt7InR5cGUiOiJyZXBvc2l0b3J5IiwibmFtZSI6InNhbWFsYmEvbXktYXBwIiwiYWN0aW9ucyI6WyJwdXNoIl19XX0.QhflHPfbd6eVF4lM9bwYpFZIV0PfikbyXuLx959ykRTBpe3CYnzs6YBK8FToVb5R47920PVLrh8zuLzdCr9t3w"}
```

## Using the signed token

Once the client has a token, it will try the registry request again with the
token placed in the HTTP `Authorization` header like so:

```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6IkJWM0Q6MkFWWjpVQjVaOktJQVA6SU5QTDo1RU42Ok40SjQ6Nk1XTzpEUktFOkJWUUs6M0ZKTDpQT1RMIn0.eyJpc3MiOiJhdXRoLmRvY2tlci5jb20iLCJzdWIiOiJCQ0NZOk9VNlo6UUVKNTpXTjJDOjJBVkM6WTdZRDpBM0xZOjQ1VVc6NE9HRDpLQUxMOkNOSjU6NUlVTCIsImF1ZCI6InJlZ2lzdHJ5LmRvY2tlci5jb20iLCJleHAiOjE0MTUzODczMTUsIm5iZiI6MTQxNTM4NzAxNSwiaWF0IjoxNDE1Mzg3MDE1LCJqdGkiOiJ0WUpDTzFjNmNueXk3a0FuMGM3cktQZ2JWMUgxYkZ3cyIsInNjb3BlIjoiamxoYXduOnJlcG9zaXRvcnk6c2FtYWxiYS9teS1hcHA6cHVzaCxwdWxsIGpsaGF3bjpuYW1lc3BhY2U6c2FtYWxiYTpwdWxsIn0.Y3zZSwaZPqy4y9oRBVRImZyv3m_S9XDHF1tWwN7mL52C_IiA73SJkWVNsvNqpJIn5h7A2F8biv_S2ppQ1lgkbw
```

This is also described in [Section 2.1 of RFC 6750: The OAuth 2.0 Authorization Framework: Bearer Token Usage](https://tools.ietf.org/html/rfc6750#section-2.1)

## Verifying the token

The registry must now verify the token presented by the user by inspecting the
claim set within. The registry will:

- Ensure that the issuer (`iss` claim) is an authority it trusts.
- Ensure that the registry identifies as the audience (`aud` claim).
- Check that the current time is between the `nbf` and `exp` claim times.
- If enforcing single-use tokens, check that the JWT ID (`jti` claim) value has
  not been seen before.
  - To enforce this, the registry may keep a record of `jti`s it has seen for
    up to the `exp` time of the token to prevent token replays.
- Check the `access` claim value and use the identified resources and the list
  of actions authorized to determine whether the token grants the required
  level of access for the operation the client is attempting to perform.
- Verify that the signature of the token is valid.

If any of these requirements are not met, the registry will return a
`403 Forbidden` response to indicate that the token is invalid.

**Note**: it is only at this point in the workflow that an authorization error
may occur. The token server should *not* return errors when the user does not
have the requested authorization. Instead, the returned token should indicate
whatever of the requested scope the client does have (the intersection of
requested and granted access). If the token does not supply proper
authorization then the registry will return the appropriate error.

At no point in this process should the registry need to call back to the
authorization server. The registry only needs to be supplied with the trusted
public keys to verify the token signatures.
