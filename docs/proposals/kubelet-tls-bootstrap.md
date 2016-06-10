<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
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

# Kubelet TLS bootstrap

Author: George Tankersley (george.tankersley@coreos.com)

## Preface

This document describes a method for a kubelet to bootstrap itself
into a TLS-secured cluster. Crucially, it automates the provision and
distribution of signed certificates.

## Overview

When a kubelet runs for the first time, it must be given TLS assets
or generate them itself. In the first case, this is a burden on the cluster
admin and a significant logistical barrier to secure Kubernetes rollouts. In
the second, the kubelet must self-sign its certificate and forfeits many of the
advantages of a PKI system. Instead, we propose that the kubelet generate a
private key and a CSR for submission to a cluster-level certificate signing
process.

## Preliminaries

We assume the existence of a functioning control plane. The
apiserver should be configured for TLS initially or possess the ability to
generate valid TLS credentials for itself. If secret information is passed in
the request (e.g. auth tokens supplied with the request or included in
ExtraInfo) then all communications from the node to the apiserver must take
place over a verified TLS connection.

Each node is additionally provisioned with the following information:

1. Location of the apiserver
2. Any CA certificates necessary to trust the apiserver's TLS certificate
3. Access tokens (if needed) to communicate with the CSR endpoint

These should not change often and are thus simple to include in a static
provisioning script.

## API Changes

### CertificateSigningRequest Object

We introduce a new API object to represent PKCS#10 certificate signing
requests. It will be accessible under:

`/apis/certificates/v1beta1/certificatesigningrequests/mycsr`

It will have the following structure:

```go
// Describes a certificate signing request
type CertificateSigningRequest struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// The certificate request itself and any additonal information.
	Spec CertificateSigningRequestSpec `json:"spec,omitempty"`

	// Derived information about the request.
	Status CertificateSigningRequestStatus `json:"status,omitempty"`
}

// This information is immutable after the request is created.
type CertificateSigningRequestSpec struct {
	// Base64-encoded PKCS#10 CSR data
	Request string `json:"request"`

	// Any extra information the node wishes to send with the request.
	ExtraInfo []string `json:"extrainfo,omitempty"`
}

// This information is derived from the request by Kubernetes and cannot be
// modified by users. All information is optional since it might not be
// available in the underlying request. This is intented to aid approval
// decisions.
type CertificateSigningRequestStatus struct {
	// Information about the requesting user (if relevant)
	// See user.Info interface for details
	Username string   `json:"username,omitempty"`
	UID      string   `json:"uid,omitempty"`
	Groups   []string `json:"groups,omitempty"`

	// Fingerprint of the public key in request
	Fingerprint string `json:"fingerprint,omitempty"`

	// Subject fields from the request
	Subject internal.Subject `json:"subject,omitempty"`

	// DNS SANs from the request
	Hostnames []string `json:"hostnames,omitempty"`

	// IP SANs from the request
	IPAddresses []string `json:"ipaddresses,omitempty"`

	Conditions []CertificateSigningRequestCondition `json:"conditions,omitempty"`
}

type RequestConditionType string

// These are the possible states for a certificate request.
const (
	Approved RequestConditionType = "Approved"
	Denied   RequestConditionType = "Denied"
)

type CertificateSigningRequestCondition struct {
	// request approval state, currently Approved or Denied.
	Type RequestConditionType `json:"type"`
	// brief reason for the request state
	Reason string `json:"reason,omitempty"`
	// human readable message with details about the request state
	Message string `json:"message,omitempty"`
	// If request was approved, the controller will place the issued certificate here.
	Certificate []byte `json:"certificate,omitempty"`
}

type CertificateSigningRequestList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`

	Items []CertificateSigningRequest `json:"items,omitempty"`
}
```

We also introduce CertificateSigningRequestList to allow listing all the CSRs in the cluster:

```go
type CertificateSigningRequestList struct {
        api.TypeMeta
        api.ListMeta

        Items []CertificateSigningRequest
}
```

## Certificate Request Process

### Node intialization

When the kubelet executes it checks a location on disk for TLS assets
(currently `/var/run/kubernetes/kubelet.{key,crt}` by default). If it finds
them, it proceeds. If there are no TLS assets, the kubelet generates a keypair
and self-signed certificate. We propose the following optional behavior:

1. Generate a keypair
2. Generate a CSR for that keypair with CN set to the hostname (or
   `--hostname-override` value) and DNS/IP SANs supplied with whatever values
   the host knows for itself.
3. Post the CSR to the CSR API endpoint.
4. Set a watch on the CSR object to be notified of approval or rejection.

### Controller response

The apiserver persists the CertificateSigningRequests and exposes the List of
all CSRs for an administrator to approve or reject.

A new certificate controller watches for certificate requests. It must first
validate the signature on each CSR and add `Condition=Denied` on
any requests with invalid signatures (with Reason and Message incidicating
such). For valid requests, the controller will derive the information in
`CertificateSigningRequestStatus` and update that object. The controller should
watch for updates to the approval condition of any CertificateSigningRequest.
When a request is approved (signified by Conditions containing only Approved)
the controller should generate and sign a certificate based on that CSR, then
update the condition with the certificate data using the `/approval`
subresource.

### Manual CSR approval

An administrator using `kubectl` or another API client can query the
CertificateSigningRequestList and update the approval condition of
CertificateSigningRequests. The default state is empty, indicating that there
has been no decision so far. A state of "Approved" indicates that the admin has
approved the request and the certificate controller should issue the
certificate. A state of "Denied" indicates that admin has denied the
request. An admin may also supply Reason and Message fields to explain the
rejection.

## kube-apiserver support

The apiserver will present the new endpoints mentioned above and support the
relevant object types.

## kube-controller-manager support

To handle certificate issuance, the controller-manager will need access to CA
signing assets. This could be as simple as a private key and a config file or
as complex as a PKCS#11 client and supplementary policy system. For now, we
will add flags for a signing key, a certificate, and a basic policy file.

## kubectl support

To support manual CSR inspection and approval, we will add support for listing,
inspecting, and approving or denying CertificateSigningRequests to kubectl. The
interaction will be similar to
[salt-key](https://docs.saltstack.com/en/latest/ref/cli/salt-key.html).

Specifically, the admin will have the ability to retrieve the full list of
pending CSRs, inspect their contents, and set their approval conditions to one
of:

1. **Approved** if the controller should issue the cert
2. **Denied** if the controller should not issue the cert

The suggested command for listing is `kubectl get csrs`. The approve/deny
interactions can be accomplished with normal updates, but would be more
conveniently accessed by direct subresource updates. We leave this for future
updates to kubectl.

## Security Considerations

### Endpoint Access Control

The ability to post CSRs to the signing endpoint should be controlled. As a
simple solution we propose that each node be provisioned with an auth token
(possibly static across the cluster) that is scoped via ABAC to only allow
access to the CSR endpoint.

### Expiration & Revocation

The node is responsible for monitoring its own certificate expiration date.
When the certificate is close to expiration, the kubelet should begin repeating
this flow until it successfully obtains a new certificate. If the expiring
certificate has not been revoked and the previous certificate request is still
approved, then it may do so using the same keypair unless the cluster policy
(see "Future Work") requires fresh keys.

Revocation is for the most part an unhandled problem in Go, requiring each
application to produce its own logic around a variety of parsing functions. For
now, our suggested best practice is to issue only short-lived certificates. In
the future it may make sense to add CRL support to the apiserver's client cert
auth.

## Future Work

- revocation UI in kubectl and CRL support at the apiserver
- supplemental policy (e.g. cluster CA only issues 30-day certs for hostnames *.k8s.example.com, each new cert must have fresh keys, ...)
- fully automated provisioning (using a handshake protocol or external list of authorized machines)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-tls-bootstrap.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
