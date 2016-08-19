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

## ClusterInfo object

First we define a set of information that identifies a cluster and how to talk to it.

This includes:

* A cluster serial number/identifier.
  * In an HA world, API servers may come and go and it is necessary to make sure we are talking to the same cluster as we thought we were talking to.
  * It looks like Federation already has this concept.  We need to back port that to non-federated clusters.
* A set of addresses for finding the cluster.
  * It is implied that all of these are equivalent and that a client can try multiple until an appropriate target is found.
  * Initially I’m proposing a flat set here.  In the future we can introduce more structure that hints to the user which addresses to try first.
* A set of root certificates to trust for this cluster.
  * We make this a set so that root certificates can be rotated.
  * These are BASE64 encoded PEM values (without `-----BEGIN CERTIFICATE-----` lines).
  This is similar to how [JWS encodes certifcates in JSON](https://www.google.com/url?q=https://tools.ietf.org/html/rfc7515%23appendix-B&sa=D&ust=1471374807953000&usg=AFQjCNFQNcmbquq3XEegKc76v5j956aSPw).
* Other root certificate options
  * Don't bother validating the certificate for debug and dev scenarios.
  * Trust the system managed list of CAs.
* The date and time this information was fetched along with hints from the server about when this information is likely to be stale.
  * This is so that a client can figure out if this information is out of date.

Any client of the cluster will want to have this information.  As the configuration of the cluster changes we need the client to keep this information up to date.  It is assumed that the information here won’t drift so fast that clients won’t be able to find *some* way to connect.

In exceptional circumstances it is possible that this information may be out of date and a client would be unable to connect to a cluster.  Consider the case where a user has kubectl set up and working well and then doesn't run kubectl for quite a while.  It is possible that over this time (a) the set of servers will have migrated so that all endpoints are now invalid or (b) the root certificates will have rotated so that the user can no longer trust any endpoint.

We serialize this into a JSON object (exact syntax still TBD -- this is a sketch).  The whitespace is inserted for readability.

```json
{
  "kind": "ClusterInfo",
  "apiVersion": "v1alpha1",
  "clusterId": "E0D87385-CE10-415F-9913-EA8388EFD80B",
  "endpoints": [
    "https://10.0.0.1",
    "https://10.0.0.2",
    "https://bastion.example.com/k8s/cluster1",
    "https://1.2.3.4",
    "https://1.2.3.5"
  ],
  "certificateAuthorities": [
    "MIIDFDCCAfygAwIBAgIJAIr/AmnKEdesMA0GCSqGSIb3DQEBBQUAMBAxDjAMBgNV
     BAMTBWpiZWRhMB4XDTE2MDgxMTIwMjc0MVoXDTQzMTIyODIwMjc0MVowEDEOMAwG
     A1UEAxMFamJlZGEwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDAiCK6
     t6LhHwojz+RQ9IVIOA2P+mL6XWSA1o6AaSMudzJRO2Gr60TFWHoz5i9+vp2ngZBR
     NOwjIc1iykvV5rFP8mNZKB3xXo6XsKXUVDP4maiyOuvywkvtnhH/HU/5oIlVPDLz
     2LWnOZd7Xip+Zgocd756p22J0oagmWien5OYHsWAlOqbut5UhfSH4gEdC/pJ+Qs8
     6N292+2A3TzJC0fX9HrcCYJEQ1v2491p8RU+G3R+IFNXPrOvuY2dHN/w3BK+Vqet
     tTKvINXCkSms6Nw9iqDjyWcRBhNg3l4j4yk5zZUmUQ2kQ0urXXZMaXSGSOZ/DLgy
     fEAPbX5rUOL1SNeRAgMBAAGjcTBvMB0GA1UdDgQWBBRncoPpkKPwuBh5oWSEK+8o
     Yo21wzBABgNVHSMEOTA3gBRncoPpkKPwuBh5oWSEK+8oYo21w6EUpBIwEDEOMAwG
     A1UEAxMFamJlZGGCCQCK/wJpyhHXrDAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEB
     BQUAA4IBAQAZpAsBflGuYSgSCZujEpHRYmP+Pl/APe37+iBioi4JgMSMKydJFvFH
     1mQaSJVc6yvNg/mqMF9drKbcX/fEdgx+25DqdXRovyvAV5/RlSRr0RA4xUaniQs0
     H0D1VUo5H9/VCH4S5CwYYPPG/1uPyKxgsRWIm3oldNiE93NseSzd65JTwg7NMISg
     9YMtbpdmDM8rrVm18QLjyjDdxEoKEfJtahl1cAw8XtoqYDczwMKITx4VoTE3gyMI
     8TKIw3YNOzjzokvEzlsscMhWryIjARjRcJ+t9kztPxB6bUEid8aOoG3OpGjpjAlp
     r4T3oIWo/YE8K2K1te/3gE/5L3QaKM2/",
    "MIIDFDCCAfygAwIBAgIJAIr/AmnKEdesMA0GCSqGSIb3DQEBBQUAMBAxDjAMBgNV
     BAMTBWpiZWRhMB4XDTE2MDgxMTIwMjc0MVoXDTQzMTIyODIwMjc0MVowEDEOMAwG
     A1UEAxMFamJlZGEwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDAiCK6
     t6LhHwojz+RQ9IVIOA2P+mL6XWSA1o6AaSMudzJRO2Gr60TFWHoz5i9+vp2ngZBR
     NOwjIc1iykvV5rFP8mNZKB3xXo6XsKXUVDP4maiyOuvywkvtnhH/HU/5oIlVPDLz
     2LWnOZd7Xip+Zgocd756p22J0oagmWien5OYHsWAlOqbut5UhfSH4gEdC/pJ+Qs8
     6N292+2A3TzJC0fX9HrcCYJEQ1v2491p8RU+G3R+IFNXPrOvuY2dHN/w3BK+Vqet
     tTKvINXCkSms6Nw9iqDjyWcRBhNg3l4j4yk5zZUmUQ2kQ0urXXZMaXSGSOZ/DLgy
     fEAPbX5rUOL1SNeRAgMBAAGjcTBvMB0GA1UdDgQWBBRncoPpkKPwuBh5oWSEK+8o
     Yo21wzBABgNVHSMEOTA3gBRncoPpkKPwuBh5oWSEK+8oYo21w6EUpBIwEDEOMAwG
     A1UEAxMFamJlZGGCCQCK/wJpyhHXrDAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEB
     BQUAA4IBAQAZpAsBflGuYSgSCZujEpHRYmP+Pl/APe37+iBioi4JgMSMKydJFvFH
     1mQaSJVc6yvNg/mqMF9drKbcX/fEdgx+25DqdXRovyvAV5/RlSRr0RA4xUaniQs0
     H0D1VUo5H9/VCH4S5CwYYPPG/1uPyKxgsRWIm3oldNiE93NseSzd65JTwg7NMISg
     9YMtbpdmDM8rrVm18QLjyjDdxEoKEfJtahl1cAw8XtoqYDczwMKITx4VoTE3gyMI
     8TKIw3YNOzjzokvEzlsscMhWryIjARjRcJ+t9kztPxB6bUEid8aOoG3OpGjpjAlp
     r4T3oIWo/YE8K2K1te/3gE/5L3QaKM2/"
  ],
  "fetchedTime": "2016-08-16T18:41:10+0000",
  "expiredTime": "2016-08-16T21:41:10+0000"
}
```

```go
type ClusterInfo struct {
  TypeMeta

  // ClusterID is a globally unique (and not necessarily human readable) ID for a cluster.
  ClusterID string

  // Endpoints are base addresses to get to the Kubernetes API server.
  Endpoints []string

  // CertificateAuthorities is a list of root certificates to trust for
  // connections to this cluster. These are PEM encoded certificates without
  // `-----BEGIN CERTIFICATE-----` lines.
  CertificateAuthorities [][]string

  // InsecureSkipTLSVerify skips the validity check for the server's
  // certificate. This will make your HTTPS connections insecure and should
  // only be used in debug and development scenarios.
  InsecureSkipTLSVerify bool

  // TrustCommonCAs instructs clients to trust the OS managed set of root
  // certificates. The server has a certificate issued by a "universally"
  // trusted CA.
  TrustCommonCAs bool

  // FetchedTime encodes the time that this information was fetched and known
  // to be good.
  FetchedTime unversioned.Time

  // ExpiredTime encodes a hint about when the client should consider this
  // information stale and re-fetch it.
  ExpiredTime unversioned.Time
}
```

**Note**: This object doesn't have regular ObjectMeta.  Since this is a singleton per API server cluster, there is no user update and no list.  Most of the members of ObjectMeta don't apply.

**TODO/Questions**:
  * Is `ClusterInfo` too general?  Do we want to rename this to `ClusterLocation` as it was in the original proposal?
  * ~~Turn this into a real Kubernetes API object with standard structure~~
  * Do we need to handle more complicated structure for specifying endpoints? Perhaps a priority?
  * ~~Does this get versioned with the rest of the API or is it a separate thing?~~
  * ~~How do we say "trust system installed roots"?  It is possible users will configure the cluster with a publicly signed certificate.~~

## Methods

Now that we know *what* we want to get to the client, the question is how.  We want to do this in as secure a way possible (as there are cryptographic keys involved) without requiring a lot of overhead in terms of information that needs to be copied around.

### Method: Out of Band

The simplest way to do this would be to simply put this object in a file and copy it around.  This is more overhead for the user, but it is easy to implement and lets users rely on existing systems to distribute configuration.

For the (currently being designed) `kubeadm` flow, the command line might look like:

```
kubeadm join --cluster-info-file=my-cluster.json
```

Note that TLS bootstrap (which establishes a way for a client to authenticate itself to the server) is a separate issue and has its own set of methods.  This command line may have a TLS bootstrap token (or config file) on the command line also.

### Method: HTTPS endpoint

If the ClusterInfo information is hosted in a trusted place of HTTPS you can just request it that way.  This will use the root certificates that are installed on the system.  It may or may not be appropriate based on the user's constraints.

```
kubeadm join --cluster-info-url="https://example/mycluster.json"
```

This is really a shorthand for someone doing something like (assuming we support stdin with `-`):

```
curl https://example.com/mycluster.json | kubeadm join --cluster-info-file=-
```

If the user requires some auth to the HTTPS server (to keep the ClusterInfo object private) that can be done in the curl command equivalent.  Or we could eventually add it to `kubeadm` directly.

### Method: JWS Token

There won’t always be a trusted external endpoint to talk to and transmitting the locator file out of band is a pain.  However, we want something a bit more secure than just hitting HTTP and trusting whatever we get back.  In this case, we assume we have the following:
  * An address for at least one of the API servers (which will implement this API).
    * This address is technically an HTTPS URL base but is often expressed as a bare domain or IP.
  * A shared secret token

An interesting aspect here is that this information is often easily obtained before the API server is configured or started.  This makes some cluster bring-up scenarios much easier.

We put some structure into the token.  It has both a token identifier and the token value, separated by a dot.  Example:

```
A81E5d4DwI.0ok9tB1QhB
```

The first part of the token is the `token-id`.  The second part is the `token-secret`.

The user experience for joining a cluster would be something like:

```
kubeadm join --token=A81E5d4DwI.0ok9tB1QhB <address>
```

**Note:** This is logically a different use of the token from TLS bootstrap.  We should look to harmonize these usages and allow the same token to play double duty.  It is likely that the bare `--token` command line flag will indicate the token be used for both ClusterInfo and TLS bootstrap.

#### Quick Primer on JWS

[JSON Web Signatures](https://tools.ietf.org/html/rfc7515) are a way to sign, serialize and verify a payload.  It supports both symmetric keys (aka shared secrets) along with asymmetric keys (aka public key infrastructure or key pairs).  The JWS is split in to 3 parts:
1. a header about how it is signed
2. the clear text payload
3. the signature.

There are a couple of different ways of encoding this data -- either as a JSON object or as a set of BASE64URL strings for including in headers or URL parameters.  In this case, we are using a shared secret and the HMAC-SHA256 signing algorithm and encoding it as a JSON object.  The popular JWT (JSON Web Tokens) specification is a type of JWS.

#### ClusterInfo Request

The API server serves up a simple plain old HTTP endpoint for this.  The client does a HTTP GET to:

```
http://<address>/api/v1alpha1/clusterinfo/?token-id=<token-id>
```

#### ClusterInfo Response

The server will look up the appropriate shared secret based on the token id.  If the server doesn't know about the token specified or there is no token specified it MUST respond with an HTTP 403 forbidden error.

If it does recognize the token it will respond with a JWS object using the flattened JWS JSON serialization syntax (JWS section 7.2.2).

The `Content-Type` header of the response MUST be `application/jose+json`. The `kid` (key id) header value MUST be the `token-id`. The `alg` MUST be `HS256` (HMAC-SHA256).

The full JWS will look something like:

```json
{
  "payload":
    "eyJ0eXBlIjoiQ2x1c3RlckxvY2F0b3IiLCJ2ZXJzaW9uIjoiMS4wIiwiZW5kcG9p
     bnRzIjpbIjEwLjAuMC4xIiwiMTAuMC4wLjIiLCJteWNsdXN0ZXIuZXhhbXBsZS5jb
     20iLCIxLjIuMy40IiwiMS4yLjMuNSJdLCJyb290Q2VydGlmaWNhdGVzIjpbIk1JSU
     RGRENDQWZ5Z0F3SUJBZ0lKQUlyL0FtbktFZGVzTUEwR0NTcUdTSWIzRFFFQkJRVUF
     NQkF4RGpBTUJnTlZCQU1UQldwaVpXUmhNQjRYRFRFMk1EZ3hNVEl3TWpjME1Wb1hE
     VFF6TVRJeU9ESXdNamMwTVZvd0VERU9NQXdHQTFVRUF4TUZhbUpsWkdFd2dnRWlNQ
     TBHQ1NxR1NJYjNEUUVCQVFVQUE0SUJEd0F3Z2dFS0FvSUJBUURBaUNLNnQ2TGhId2
     9qeitSUTlJVklPQTJQK21MNlhXU0ExbzZBYVNNdWR6SlJPMkdyNjBURldIb3o1aTk
     rdnAybmdaQlJOT3dqSWMxaXlrdlY1ckZQOG1OWktCM3hYbzZYc0tYVVZEUDRtYWl5
     T3V2eXdrdnRuaEgvSFUvNW9JbFZQREx6MkxXbk9aZDdYaXArWmdvY2Q3NTZwMjJKM
     G9hZ21XaWVuNU9ZSHNXQWxPcWJ1dDVVaGZTSDRnRWRDL3BKK1FzODZOMjkyKzJBM1
     R6SkMwZlg5SHJjQ1lKRVExdjI0OTFwOFJVK0czUitJRk5YUHJPdnVZMmRITi93M0J
     LK1ZxZXR0VEt2SU5YQ2tTbXM2Tnc5aXFEanlXY1JCaE5nM2w0ajR5azV6WlVtVVEy
     a1EwdXJYWFpNYVhTR1NPWi9ETGd5ZkVBUGJYNXJVT0wxU05lUkFnTUJBQUdqY1RCd
     k1CMEdBMVVkRGdRV0JCUm5jb1Bwa0tQd3VCaDVvV1NFSys4b1lvMjF3ekJBQmdOVk
     hTTUVPVEEzZ0JSbmNvUHBrS1B3dUJoNW9XU0VLKzhvWW8yMXc2RVVwQkl3RURFT01
     Bd0dBMVVFQXhNRmFtSmxaR0dDQ1FDSy93SnB5aEhYckRBTUJnTlZIUk1FQlRBREFR
     SC9NQTBHQ1NxR1NJYjNEUUVCQlFVQUE0SUJBUUFacEFzQmZsR3VZU2dTQ1p1akVwS
     FJZbVArUGwvQVBlMzcraUJpb2k0SmdNU01LeWRKRnZGSDFtUWFTSlZjNnl2TmcvbX
     FNRjlkcktiY1gvZkVkZ3grMjVEcWRYUm92eXZBVjUvUmxTUnIwUkE0eFVhbmlRczB
     IMEQxVlVvNUg5L1ZDSDRTNUN3WVlQUEcvMXVQeUt4Z3NSV0ltM29sZE5pRTkzTnNl
     U3pkNjVKVHdnN05NSVNnOVlNdGJwZG1ETThyclZtMThRTGp5akRkeEVvS0VmSnRha
     GwxY0F3OFh0b3FZRGN6d01LSVR4NFZvVEUzZ3lNSThUS0l3M1lOT3pqem9rdkV6bH
     NzY01oV3J5SWpBUmpSY0ordDlrenRQeEI2YlVFaWQ4YU9vRzNPcEdqcGpBbHByNFQ
     zb0lXby9ZRThLMksxdGUvM2dFLzVMM1FhS00yLyJdfQ",
  "protected":"eyJhbGciOiJIUzI1NiIsImtpZCI6IkE4MUU1ZDREd0kifQ",
  "signature":"vat_c1aTytSLVkPw9wWhven9GtLBSLFzJViIq5scLDA"
}
```

If we Base64 decode the `protected` member we get:

```json
{
  "alg":"HS256",
  "kid":"A81E5d4DwI"
}
```

The payload is the ClusterInfo JSON object as described above.  Astute readers will note that the actual root certificates are double Base64 encoded.  It is a good thing that efficiency isn’t critical here.

*Implementation Note:* there looks to be a decent library for dealing with JWS: https://github.com/square/go-jose.

#### Method: ClusterInfo refresh

The API server will serve up a version of this endpoint at `https://<address>/api/v1alpha1/clusterinfo` (note the `https`).  Assuming the user authenticates to (and verifies identity of) the API server, this returns the `ClusterInfo` object directly.  This is used by clients that already have communication with the cluster to update their cached copy of the ClusterInfo object.  It is expected that the client will respect the `expiredTime` member of the object and refresh before the information is expired if at all possible.

Implementation note: We'll probably want to jitter this to avoid a stampede.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/super-simple-discovery-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
