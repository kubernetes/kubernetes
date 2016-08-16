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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Super Simple Discovery API

## Overview

It is surprisingly hard to figure out how to talk to a Kubernetes cluster.  Not only do clients need to know where to look on the network, they also need to identify the set of root certificates to trust when talking to that endpoint.

This presents a set of problems:
* It should be super easy for users to configure client systems with a minimum of effort `kubectl` or `kubeadm init` (or other client systems).
  * Establishing this should be doable even in the face of nodes and master components booting out of order.
  * We should have mechanisms that don't require users to ever have to manually manage certificate files.
* Over the life of the cluster this information could change and client systems should be able to adapt.

While this design is mainly being created to help `kubeadm` possible, these problems aren't isolated there and can be used outside of the kubeadm context.

Mature organizations should be able to distribute and manage root certificates out of band of Kubernetes installations.  In that case, clients will defer to corporation wide system installed root certificates or root certificates distributed through other means.  However, for smaller and more casual users distributing or obtaining certificates represents a challenge.

Similarly, mature organizations will be able to rely on a centrally managed DNS system to distribute the location of a set of API servers and keep those names up to date over time.  Those DNS servers will be managed for high availability.

With that in mind, the proposals here will devolve into simply using DNS names that are validated with system installed root certificates.

## Cluster Location information

First we define a set of information that identifies a cluster and how to talk to it.

While we could define a new format for communicating the set of information needed here, we'll start by using the standard [`kubeconfig`](http://kubernetes.io/docs/user-guide/kubeconfig-file/) file format.

It is expected that the `kubeconfig` file will have a single unnamed `Cluster` entry.  Other information (especially authentication secrets) must be omitted.

### Evolving kubeconfig

In the future we look forward to enhancing `kubeconfig` to address some issues.  These are out of scope for this design.  Some of this is covered in [#30395](https://github.com/kubernetes/kubernetes/issues/30395).  

Additions include:

* A cluster serial number/identifier.
  * In an HA world, API servers may come and go and it is necessary to make sure we are talking to the same cluster as we thought we were talking to.
* A _set_ of addresses for finding the cluster.
  * It is implied that all of these are equivalent and that a client can try multiple until an appropriate target is found.
  * Initially I’m proposing a flat set here.  In the future we can introduce more structure that hints to the user which addresses to try first.
* A _set_ of root certificates to trust for this cluster.
  * We make this a set so that root certificates can be rotated.
* Other root certificate options
  * Trust the system managed list of CAs.

### Client caching and update

**This is to be implemented in a later phase**

Any client of the cluster will want to have this information.  As the configuration of the cluster changes we need the client to keep this information up to date.  It is assumed that the information here won’t drift so fast that clients won’t be able to find *some* way to connect.

In exceptional circumstances it is possible that this information may be out of date and a client would be unable to connect to a cluster.  Consider the case where a user has kubectl set up and working well and then doesn't run kubectl for quite a while.  It is possible that over this time (a) the set of servers will have migrated so that all endpoints are now invalid or (b) the root certificates will have rotated so that the user can no longer trust any endpoint.

## Methods

Now that we know *what* we want to get to the client, the question is how.  We want to do this in as secure a way possible (as there are cryptographic keys involved) without requiring a lot of overhead in terms of information that needs to be copied around.

### Method: Out of Band

The simplest way to do this would be to simply put this object in a file and copy it around.  This is more overhead for the user, but it is easy to implement and lets users rely on existing systems to distribute configuration.

For the `kubeadm` flow, the command line might look like:

```
kubeadm join --cluster-info-file=my-cluster.yaml
```

Note that TLS bootstrap (which establishes a way for a client to authenticate itself to the server) is a separate issue and has its own set of methods.  This command line may have a TLS bootstrap token (or config file) on the command line also.

### Method: HTTPS endpoint

If the ClusterInfo information is hosted in a trusted place via HTTPS you can just request it that way.  This will use the root certificates that are installed on the system.  It may or may not be appropriate based on the user's constraints.

```
kubeadm join --cluster-info-url="https://example/mycluster.yaml"
```

This is really a shorthand for someone doing something like (assuming we support stdin with `-`):

```
curl https://example.com/mycluster.json | kubeadm join --cluster-info-file=-
```

If the user requires some auth to the HTTPS server (to keep the ClusterInfo object private) that can be done in the curl command equivalent.  Or we could eventually add it to `kubeadm` directly.

### Method: Token

There won’t always be a trusted external endpoint to talk to and transmitting
the locator file out of band is a pain.  However, we want something more secure
than just hitting HTTP and trusting whatever we get back.  In this case, we
assume we have the following:

  * An address for at least one of the API servers (which will implement this API).
    * This address is technically an HTTPS URL base but is often expressed as a bare domain or IP.
  * A shared secret token

An interesting aspect here is that this information is often easily obtained before the API server is configured or started.  This makes some cluster bring-up scenarios much easier.

The user experience for joining a cluster would be something like:

```
kubeadm join --token=ae23dc.faddc87f5a5ab458 <address>
```

**Note:** This is logically a different use of the token from TLS bootstrap.  We harmonize these usages and allow the same token to play double duty.

#### Implementation Flow

`kubeadm` will implement the following flow:

* `kubeadm` connects to the API server address specified over TLS.  As we don't yet have a root certificate to trust, this is an insecure connection and the server certificate is not validated.
  * Implementation note: the API server doesn't have to expose a new and special insecure HTTP endpoint.
* `kubeadm` requests a ConfigMap containing the kubeconfig file defined above.
  * This ConfigMap exists at a well known URL: `https://<server>//api/v1/namespaces/kube-public/configmaps/cluster-info`
  * `kubeadm` uses a the token to create a short lived JWT that is passed in the HTTP Authentication header.  We can't just pass the token as we don't yet trust the server.
* The API server returns the ConfigMap with the kubeconfig contents as normal
  * An extra header (`X-Kubernetes-JWS`) is added with a signature using the same token as used for authentication.
* `kubeadm` verifies the JWS and can now trust the server.  Further
  communication is simpler as the CA certificate in the kubeconfig file can be
  trusted.


#### NEW: Token Structure

To first make this work, we put some structure into the token.  It has both a token identifier and the token value, separated by a dot.  Example:

```
ae23dc.faddc87f5a5ab458
```

The first part of the token is the `token-id`.  The second part is the `token-secret`.  By having a token identifier, we make it easier to specify *which* token you are talking about without sending the token itself in the clear.

To support this new type of token, we will update the CSV token authenticator to recognize these types of tokens and index them by `token-id`.  Tokens that do not conform to this method will continue to work but won't be usable using the JWT based authentication mechanism defined below.

#### Quick Primer on JWS and JWT

[JSON Web Signatures](https://tools.ietf.org/html/rfc7515) are a way to sign, serialize and verify a payload.  It supports both symmetric keys (aka shared secrets) along with asymmetric keys (aka public key infrastructure or key pairs).  The JWS is split in to 3 parts:
1. a header about how it is signed
2. the clear text payload
3. the signature.

There are a couple of different ways of encoding this data -- either as a JSON object or as a set of BASE64URL strings for including in headers or URL parameters.  In this case, we are using a shared secret and the HMAC-SHA256 signing algorithm and encoding it as a JSON object.  The popular JWT (JSON Web Tokens) specification is a type of JWS.

The JWS specification [describes how to encode](https://tools.ietf.org/html/rfc7515#appendix-F) "detached content".  In this way the signature is calculated as normal but the content isn't included in the signature.

#### NEW: JWT authentication

To avoid passing an unencoded token we will instead create a short lived JWT based on the token.  The JWT will be constructed as so:

* **Header**
  * **typ**: `JWT`. REQUIRED.
  * **alg**: `HS256` for HMAC SHA-256. REQUIRED.
  * **kid**: The token-id. REQUIRED.
* **Claims**
  * **sub**: The token-id. REQUIRED.
  * **aud**: The long term globally unique cluster identifier.  This is not implemented yet so is OPTIONAL.
  * **exp**: An expiration time for this JWT.  This must be no more than 2 minutes in the future. REQUIRED.
  * **iat**: The time that the JWT was issued.  OPTIONAL.

**Question:** The 2 minute value is picked to be short enough to discourage replay attacks without introducing the complexity of tracking nonces.  If we mandated the use of nonces, we'd have to track which nonces were used across HA instances.  Is there a better value than 2 minutes?  If we see clock skew greater or connectivity problems greater than 2 minutes we'll probably have other problems.

#### NEW: `kube-public` namespace

Kubernetes ConfigMaps are per-namespace and are generally only visible to principals that have read access on that namespace.  To create a config map that *everyone* can see, we introduce a new `kube-public` namespace.  This namespace, by convention, is readable by all authenticated users.

#### NEW: JWS content signing

When the JWT flow is used, we go one step further and sign the payload of the response using a JWS with detached content.

We compute the JWS as normal (using the token provided in the JWT and `HS256`) and then remove the middle section that contains the BASE64URL encoded content.  This shortened JWS (which only has the JWS header and signature) is then put in a new HTTP header.  This header is `X-Kubernetes-JWS`.

We use a "detached content" JWS to ease implementation.  This can be implemented a simple HTTP request filter without having to modify the subtle content negotiation mechanisms in the API server.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/super-simple-discovery-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
