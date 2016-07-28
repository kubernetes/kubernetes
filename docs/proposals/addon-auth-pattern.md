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

When a cluster service, such as Dashboard is being deployed:
1. Dashboard deployment and service are is created, in namespace `kube-system`.  Service is called `dashboard`.

When a user wants to access a service:
1. User runs `kubectl proxy -p 8001 &`.
  -  This causes kubectl to serve HTTP on `localhost:8001`, and to reverse proxy that to the users current cluster, using TLS and the users
     currently configured authentication method.
1. User enters this into their browser navigation bar: `http://localhost:8001/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard/#/workload`
  - This causes the API server to proxy a single HTTP request.
  - The effect of the two proxies is that an HTTP request on localhost:8001 becomes an HTTP request for `/#/workloads` on port 80 on the pod IP of one of the Pods that backs services `dashboard` in namespace `kube-system`.  So far so good.
1. The Dashboard sees the request.  It does not know what users is making the request.
1. The response is sent back from the Pod to the Apiserver.
1. The response is sent back to the user.
1. The user sees the Kubernetes Dashboard page.

The problem with this is that the Dashboard does not know which user is accessing it, so it cannot support multiple users with different capabilities.

### Proposed Pattern

The proposed pattern has the following key improvements over the current pattern:
- a header is injected which tells the Dashboard pod the user who is reaching it.
- the header can be validated easily as being signed by the apiserver, which is trusted to authenticate users

When a cluster service, such as Dashboard is being deployed:
1. Dashboard deployment and service are is created, in namespace `kube-system`.  Service is called `dashboard`.
  - *NEW* The service has annotation `alpha.service-proxy.kubernetes.io/proxy-authentication` set to `true`.
    Only services that want the authentication header injected set this option.

When a user wants to access a service:
1. User runs `kubectl proxy -p 8001 &`.
1. User enters this into their browser navigation bar: `http://localhost:8001/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard/#/workload`
  - This causes the API server to proxy a single HTTP request.
  - The effect of the two proxies is that an HTTP request on localhost:8001 becomes an HTTP request for `/#/workloads` on port 80 on the pod IP of one of the Pods that backs services `dashboard` in namespace `kube-system`.  So far so good.
1. *NEW* When the API server proxies a request to a service with annotation `alpha.service-proxy.kubernetes.io/proxy-authentication`, it does the following extra steps:
    1. Injects a `X-Remote-User:` header with the verified username of the user. (Similar to [Openshift Request Header Authentication](https://docs.openshift.com/enterprise/3.0/admin_guide/configuring_authentication.html#RequestHeaderIdentityProvider) )
    1. Presents a client certificate which identifies it as being the API server, and thus a trusted
       verifier of identities. (see https://github.com/kubernetes/kubernetes/pull/26634 for client cert proxy support).
1. The Dashboard sees the request.
  - *NEW*  It verifies the Client Cert is signed by the same Root CA as is in its kubeconfig file, so it trusts that the connection is from the apiserver.
  - *NEW* The dashboard sees the `X-Remote-User` header and can use this in its own authorization logic.
1. The response is sent back from the Pod to the Apiserver.
1. The response is sent back to the user.
1. The user sees the Kubernetes Dashboard page.

The Dashboard does not need a valid SSL cert for this to work.  Since the apiserver presents a client
cert, it can verify the identity of the apiserver.  (Verify that someone cannot MITM the connection and
present a modified request to a different service?)

Revocation:

- if a user has access to the cluster revoked, then the next time an HTTP request is sent from the browser, it will be blocked by the apiserver,
and not forwarded.

#### Considerations

- Need rate limit per user in the apiserver which is fairly low bandwidth.
- Assumes HTTP.  Not suitable for non-HTTP protocols like Redis.
- Need to limit length of HTTP requests (no hanging gets that complete a month later or a long lasting SPDY tunnel), so that when a credential is revoked, the access will stop
- Evaluate MITM risks if there is no server cert for dashboard.
- Need to also delete this header if the original request sets it.
- Destination needs to be able to distinguish between an apiserver which added the header, and an old version of
  apiserver which did not add the header, and the source added the header.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/addon-auth-pattern.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
