# Kubelet Authentication / Authorization

Author: Jordan Liggitt (jliggitt@redhat.com)

## Overview

The kubelet exposes endpoints which give access to data of varying sensitivity,
and allow performing operations of varying power on the node and within containers.
There is no built-in way to limit or subdivide access to those endpoints,
so deployers must secure the kubelet API using external, ad-hoc methods.

This document proposes a method for authenticating and authorizing access
to the kubelet API, using interfaces and methods that complement the existing
authentication and authorization used by the API server.

## Preliminaries

This proposal assumes the existence of:

* a functioning API server
* the SubjectAccessReview and TokenReview APIs

It also assumes each node is additionally provisioned with the following information:

1. Location of the API server
2. Any CA certificates necessary to trust the API server's TLS certificate
3. Client credentials authorized to make SubjectAccessReview and TokenReview API calls

## API Changes

None

## Kubelet Authentication

Enable starting the kubelet with one or more of the following authentication methods:

* x509 client certificate
* bearer token
* anonymous (current default)

For backwards compatibility, the default is to enable anonymous authentication.

### x509 client certificate

Add a new `--client-ca-file=[file]` option to the kubelet.
When started with this option, the kubelet authenticates incoming requests using x509
client certificates, validated against the root certificates in the provided bundle.
The kubelet will reuse the x509 authenticator already used by the API server.

The master API server can already be started with `--kubelet-client-certificate` and
`--kubelet-client-key` options in order to make authenticated requests to the kubelet.

### Bearer token

Add a new `--authentication-token-webhook=[true|false]` option to the kubelet.
When true, the kubelet authenticates incoming requests with bearer tokens by making
`TokenReview` API calls to the API server.

The kubelet will reuse the webhook authenticator already used by the API server, configured
to call the API server using the connection information already provided to the kubelet.

To improve performance of repeated requests with the same bearer token, the
`--authentication-token-webhook-cache-ttl` option supported by the API server
would be supported.

### Anonymous

Add a new `--anonymous-auth=[true|false]` option to the kubelet.
When true, requests to the secure port that are not rejected by other configured
authentication methods are treated as anonymous requests, and given a username
of `system:anonymous` and a group of `system:unauthenticated`.

## Kubelet Authorization

Add a new `--authorization-mode` option to the kubelet, specifying one of the following modes:
* `Webhook`
* `AlwaysAllow` (current default)

For backwards compatibility, the authorization mode defaults to `AlwaysAllow`.

### Webhook

Webhook mode converts the request to authorization attributes, and makes a `SubjectAccessReview`
API call to check if the authenticated subject is allowed to make a request with those attributes.
This enables authorization policy to be centrally managed by the authorizer configured for the API server.

The kubelet will reuse the webhook authorizer already used by the API server, configured
to call the API server using the connection information already provided to the kubelet.

To improve performance of repeated requests with the same authenticated subject and request attributes,
the same webhook authorizer caching options supported by the API server would be supported:

* `--authorization-webhook-cache-authorized-ttl`
* `--authorization-webhook-cache-unauthorized-ttl`

### AlwaysAllow

This mode allows any authenticated request.

## Future Work

* Add support for CRL revocation for x509 client certificate authentication (http://issue.k8s.io/18982)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-auth.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
