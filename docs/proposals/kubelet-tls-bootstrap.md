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

`/api/vX/certificaterequests/mycsr`

It will have the following structure:

```go
// Describes a certificate signing request
type CertificateSigningRequest struct {
        api.TypeMeta         `json:",inline"`
        api.ObjectMeta       `json:"metadata,omitempty"`

        // Specifies the behavior of the CSR
        Spec CertificateSigningRequestSpec

        // Most recently observed status of the CSR
        Status CertificateSigningRequestStatus
}

type CertificateSigningRequestSpec struct {
        // Raw PKCS#10 CSR data
        CertificateRequest []byte

        // Fingerprint of the public key that signed the CSR
        Fingerprint string

        // Subject fields from the CSR
        Subject pkix.Name

        // DNS SANs from the CSR
        Hostnames []string

        // IP SANs from the CSR
        IPAddresses []string

        // Extra information the node wishes to send with the request
        ExtraInfo []string
}

type CertificateSigningRequestStatus struct {
        // Indicates whether CSR has a response yet. Default is Unknown. Status
        // is True for approval and False for rejections.
        Status api.ConditionStatus

        // If CSR was rejected, these contain the reason why (if any was supplied).
        Reason string
        Message string

        // If CSR was approved, this contains the issued certificate.
        Certificate []byte
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
and self-signed certificate. We propose the following optional fallback behavior:

1. Generate a keypair
2. Generate a CSR for that keypair with CN set to the hostname (or
   `--hostname-override` value) and DNS/IP SANs supplied with whatever values
   the host knows for itself.
3. Post the CSR to the CSR API endpoint.
4. Set a watch on the CSR object to be notified of approval or rejection.

### Controller response
The apiserver must first validate the signature on the raw CSR data and reject
requests featuring invalid CSRs. It then persists the
CertificateSigningRequests and exposes the List of all CSRs for an
administrator to approve or reject. The apiserver should watch for updates the
Status field of any CertificateSigningRequest. When a CSR is approved
(signified by Status changing from Unknown to True) the apiserver should
generate and sign the certificate, then update the
CertificateSigningRequestStatus with the new data.

### Manual CSR approval
An administrator using `kubectl` or another API client can query the
CertificateSigningRequestList and update the status of
CertificateSigningRequests. The default Status is Unknown, indicating that
there has been no decision so fare. A Status of True indicates that the admin
has approved the request and the apiserver should issue the certificate. A
Status of False indicates that the admin has denied the request. An admin may
also supply Reason and Message fields to explain the rejection.

## kube-apiserver support (CA assets)
So that the apiserver can handle certificate issuance on its own, it will need
access to CA signing assets. This could be as simple as a private key and a
config file or as complex as a PKCS#11 client and supplementary policy system.
For now, we will add flags for a signing key, a certificate, and a basic config
file.

## kubectl support
To support manual CSR inspection and approval, we will add support for listing,
inspecting, and approving/rejecting CertificateSigningRequests to kubectl. The
interface will be similar to
[salt-key](https://docs.saltstack.com/en/latest/ref/cli/salt-key.html).

Specifically, the admin will have the ability to retrieve the full list of
active CSRs, inspect their contents, and set their statuses to one of:

1. **approved** if the apiserver should issue the cert
2. **rejected** if the apiserver should not issue the cert

The suggested commands are `kubectl get certificates`, `kubectl approve <csr>`
and `kubectl reject <csr>`. For the reject subcommand, the admin will also be
able to supply Reason and Message fields via additional flags.

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
certificate has not been revoked then it may do so using the same keypair
unless the cluster policy (see "Future Work") requires fresh keys.

Revocation is for the most part an unhandled problem in Go, requiring each
application to produce its own logic around a variety of parsing functions. For
now, our suggested best practice is to issue only short-lived certificates. In
the future it may make sense to add CRL support to the apiserver's client cert
auth.

## Future Work
- revocation UI in kubectl and CRL support at the apiserver
- supplemental policy (e.g. cluster CA only issues 30-day certs for hostnames *.k8s.example.com, each new cert must have fresh keys, ...)
- fully automated provisioning (using a handshake protocol or external list of authorized machines)
