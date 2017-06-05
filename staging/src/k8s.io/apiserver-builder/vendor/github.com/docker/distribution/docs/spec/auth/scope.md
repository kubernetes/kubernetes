<!--[metadata]>
+++
title = "Token Scope Documentation"
description = "Describes the scope and access fields used for registry authorization tokens"
keywords = ["registry, on-prem, images, tags, repository, distribution, advanced, access, scope"]
[menu.main]
parent="smn_registry_ref"
+++
<![end-metadata]-->

# Docker Registry Token Scope and Access

Tokens used by the registry are always restricted what resources they may
be used to access, where those resources may be accessed, and what actions
may be done on those resources. Tokens always have the context of a user which
the token was originally created for. This document describes how these
restrictions are represented and enforced by the authorization server and
resource providers.

## Scope Components

### Subject (Authenticated User)

The subject represents the user for which a token is valid. Any actions
performed using an access token should be considered on behalf of the subject.
This is included in the `sub` field of access token JWT. A refresh token should
be limited to a single subject and only be able to give out access tokens for
that subject.

### Audience (Resource Provider)

The audience represents a resource provider which is intended to be able to
perform the actions specified in the access token. Any resource provider which
does not match the audience should not use that access token. The audience is
included in the `aud` field of the access token JWT. A refresh token should be
limited to a single audience and only be able to give out access tokens for that
audience.

### Resource Type

The resource type represents the type of resource which the resource name is
intended to represent. This type may be specific to a resource provider but must
be understood by the authorization server in order to validate the subject
is authorized for a specific resource.

#### Example Resource Types

 - `repository` - represents a single repository within a registry. A
repository may represent many manifest or content blobs, but the resource type
is considered the collections of those items. Actions which may be performed on
a `repository` are `pull` for accessing the collection and `push` for adding to
it.

### Resource Name

The resource name represent the name which identifies a resource for a resource
provider. A resource is identified by this name and the provided resource type.
An example of a resource name would be the name component of an image tag, such
as "samalba/myapp".

### Resource Actions

The resource actions define the actions which the access token allows to be
performed on the identified resource. These actions are type specific but will
normally have actions identifying read and write access on the resource. Example
for the `repository` type are `pull` for read access and `push` for write
access.

## Authorization Server Use

Each access token request may include a scope and an audience. The subject is
always derived from the passed in credentials or refresh token. When using
a refresh token the passed in audience must match the audience defined for
the refresh token. The audience (resource provider) is provided using the
`service` field. Multiple resource scopes may be provided using multiple `scope`
fields on the `GET` request. The `POST` request only takes in a single
`scope` field but may use a space to separate a list of multiple resource
scopes.

### Resource Scope Grammar

```
scope                   := resourcescope [ ' ' resourcescope ]*
resourcescope           := resourcetype  ":" resourcename  ":" action [ ',' action ]*
resourcetype            := /[a-z]*/
resourcename            := component [ '/' component ]*
action                  := /[a-z]*/
component               := alpha-numeric [ separator alpha-numeric ]*
alpha-numeric           := /[a-z0-9]+/
separator               := /[_.]|__|[-]*/
```
Full reference grammar is defined
(here)[https://godoc.org/github.com/docker/distribution/reference]. Currently
the scope name grammar is a subset of the reference grammar without support
for hostnames.

## Resource Provider Use

Once a resource provider has verified the authenticity of the scope through
JWT access token verification, the resource provider must ensure that scope
satisfies the request. The resource provider should match the given audience
according to name or URI the resource provider uses to identify itself. Any
denial based on subject is not defined here and is up to resource provider, the
subject is mainly provided for audit logs and any other user-specific rules
which may need to be provided but are not defined by the authorization server.

The resource provider must ensure that ANY resource being accessed as the
result of a request has the appropriate access scope. Both the resource type
and resource name must match the accessed resource and an appropriate action
scope must be included.

When appropriate authorization is not provided either due to lack of scope
or missing token, the resource provider to return a `WWW-AUTHENTICATE` HTTP
header with the `realm` as the authorization server, the `service` as the
expected audience identifying string, and a `scope` field for each required
resource scope to complete the request.

## JWT Access Tokens

Each JWT access token may only have a single subject and audience but multiple
resource scopes. The subject and audience are put into standard JWT fields
`sub` and `aud`. The resource scope is put into the `access` field. The
structure of the access field can be seen in the
[jwt documentation](jwt.md).

## Refresh Tokens

A refresh token must be defined for a single subject and audience. Further
restricting scope to specific type, name, and actions combinations should be
done by fetching an access token using the refresh token. Since the refresh
token is not scoped to specific resources for an audience, extra care should
be taken to only use the refresh token to negotiate new access tokens directly
with the authorization server, and never with a resource provider.

