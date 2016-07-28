<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/addon-auth-pattern.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Pattern for Addon Authentication

This document proposes a reference pattern for cluster addons to authenticate users, and suggests two new 
features that enable this pattern.

### Goals

- Programs which run on the kubernetes cluster, and which provide services to users of the cluster,
  can authenticate those users.
  - Can implement this without the need to write much authentication code, allowing ease of development/extending kubernetes.
- Programs that do not need to handle authentication credentials do not have to, thus allowing a "service admin"
  to maintain a service, without that service admin ever getting access to other users authentication creds,
  which could then be replayed against other services.
- To keep authentication and authorization separate, since this improves composability.
- Allow users to reuse the cluster credentials in their `~/.kube/config`, so they do not need
  creds per cluster
- Do not use basic-auth creds in browsers, which are subject to CSRF attacks.
- Do not use client certs in browsers, which are hard to install.

The example used in this pattern is the kubernetes Dashboard.

### Current pattern which has problems

The following pattern is in use today.

1. Dashboard is running with service `dashboard` in namespace `kube-system`.
1. User runs `kubectl proxy -p 8001 &`.
  -  This causes kubectl to serve HTTP on `localhost:8001`, and to reverse proxy that to the users current cluster, using TLS and the users
     currently configured authentication method.
1. User enters this into their browser navigation bar: `http://localhost:8001/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard/#/workload`
  - This causes the API server to proxy a single HTTP request.  The effect of the two proxies is that an HTTP request on localhost:8001 becomes an HTTP request for `/workloads` on port 80 on the pod IP of one of the Pods that backs services `dashboard` in namespace `kube-system`.  So far so good.
1. The response is sent back from the Pod to the Apiserver.
1. The response is sent back to the users.
1. The user sees the Kubernetes Dashboard.

The problem with this is that the Dashboard does not know which user is accessing it, so it cannot support multiple users with different capabilities.

### Proposed Pattern

The proposed pattern has the following key improvements over the current pattern:
- a header is injected which tells the Dashboard pod the user who is reaching it.
- the header can be validated easily as being signed by the apiserver, which is trusted to authenticate users

1. Dashboard is running with service `dashboard` in namespace `kube-system`.
1. User runs `kubectl proxy -p 8001 &`.
  -  Same as before.
1. User enters this into their browser navigation bar: `http://localhost:8001/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard/?add-auth=true#/workload`
  - Difference is the addition of the `add-auth=true` query parameter, which is handled by the APIserver.  When the API server sees this, it:
    1. checks for a cached token
    1. if not found, generates and signs a token which attests the identity of the user.
    1. injects a header `Authorization: Bearer $TOKEN` into the HTTP request.
1. The Dashboard sees the request.
1. The dashboard sees the Authorization header and verifies the signature of the token, and extracts the username from one of the claims of the token.
1. The dashboard uses the username as needed in its authorization logic (e.g. put into `Impersonate-User` header when doing actions on users behalf.
1. The dashboard sends a response back to the user.
1. The user sees the Kubernetes Dashboard.

Properties of this token:
- It is a JWT
- The sub (subject) claim gives the kubernetes username of the user. The exact format of these is up to how the cluster admin setup apiserver auth in the cluster.
- The exp (expiration) claim is some short amount of time, say a minute.  This prevents replay.
- The aud (audience) claim contains the namespace and name of the service that the request is being forwarded to.  This allows one service to ensure that it is not getting tokens intended for another service (in other words, it prevents Dashboard from replaying tokens to another service in the cluster)
- The token is signed with the private key of the apiserver.

A thing that wants to verify the JWT should do the normal JWT verifying things, which include:
- check expiration
- check signature.  In this case, the public key is already available inside the cert for the apiserver, which is automatically provided to pods
  running on the cluster by the service account controller.
- Check that the aud claim matches the service that this pod thinks it belongs in (How it does this is TDB).

Revokation:

- if a user has access to the cluster revoked, then the next time an HTTP request is sent from the browser, it will be blocked by the apiserver,
and not forwarded.

This solves the first problem, but it does mean that Dashboard and other add-ons need to correctly handle these special JWTs.

### Proposed Pattern v2

This pattern builds on the previous one by having a forward proxy on the Dashboard side which verifies the JWT so that the application (e.g. Dashboard)
does not have to implement JWT verification.

In this patterm the Dashboard container runs in a pod that also has a sidecar.  That sidecar runs a forward proxy which verifies that requests
have a proper `Authentication: Bearer $TOKEN` header.  The forward proxy listens on the pod IP, and forwards to port 80 on localhost.  The dashboard
listens on localhost:80.  The forward proxy gives a 403 if the auth is wrong.  If it is good, it replaces the Authentication header with an `X-Authorized-User: $USERNAME` header and forwards.  The dashboard does not need to verify the JWT.  It just looks for the `X-Authenticated-User` header.

The complete flow is:

1. Dashboard is running with service `dashboard` in namespace `kube-system`.
1. User runs `kubectl proxy -p 8001 &`.  
  - same as previous
1. User enters this into their browser navigation bar: `http://localhost:8001/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard/?add-auth=true#/workload`
  - same as previous
1. The sidecar in the Dashboard pod sees the request and verifies the `Authentication` header.
  - it replaces `Authentication` with `X-Authenticated-User`
  - it then forwards to localhost:80
1. The Dashboard sees the request.
  - same as previous
1. The dashboard uses the username as needed in its authorization logic
  - same as previous
1. The dashboard sends a response back to the user.
  - same as previous
1. The user sees the Kubernetes Dashboard.
  - same as previous

The forward proxy could be implemented in `kubectl`, e.g. `kubectl forward-proxy --listen-on-pod-ip --listen-port 80 -forward-ip 127.0.0.1 --forward-port 80`.  The forward proxy could periodically list services in the namespace to determin what services the pod belongs to so that it can verify `aud` claims.

#### Considerations

- Need rate limit per user in the apiserver which is fairly low bandwidth.
- Assumes HTTP.  Not suitable for non-HTTP protocols like Redis.
- Need to limit length of HTTP requests (no hanging gets that complete a month later or a long lasting SPDY tunnel), so that when a credential is revoked, the access will stop

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/addon-auth-pattern.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
