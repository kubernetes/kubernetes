/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:prerelease-lifecycle-gen:introduced=1.26
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterTrustBundle is a cluster-scoped container for X.509 trust anchors
// (root certificates).
//
// ClusterTrustBundle objects are considered to be readable by any authenticated
// user in the cluster, because they can be mounted by pods using the
// `clusterTrustBundle` projection.  All service accounts have read access to
// ClusterTrustBundles by default.  Users who only have namespace-level access
// to a cluster can read ClusterTrustBundles by impersonating a serviceaccount
// that they have access to.
//
// It can be optionally associated with a particular assigner, in which case it
// contains one valid set of trust anchors for that signer. Signers may have
// multiple associated ClusterTrustBundles; each is an independent set of trust
// anchors for that signer. Admission control is used to enforce that only users
// with permissions on the signer can create or modify the corresponding bundle.
type ClusterTrustBundle struct {
	metav1.TypeMeta `json:",inline"`

	// metadata contains the object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec contains the signer (if any) and trust anchors.
	Spec ClusterTrustBundleSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
}

// ClusterTrustBundleSpec contains the signer and trust anchors.
type ClusterTrustBundleSpec struct {
	// signerName indicates the associated signer, if any.
	//
	// In order to create or update a ClusterTrustBundle that sets signerName,
	// you must have the following cluster-scoped permission:
	// group=certificates.k8s.io resource=signers resourceName=<the signer name>
	// verb=attest.
	//
	// If signerName is not empty, then the ClusterTrustBundle object must be
	// named with the signer name as a prefix (translating slashes to colons).
	// For example, for the signer name `example.com/foo`, valid
	// ClusterTrustBundle object names include `example.com:foo:abc` and
	// `example.com:foo:v1`.
	//
	// If signerName is empty, then the ClusterTrustBundle object's name must
	// not have such a prefix.
	//
	// List/watch requests for ClusterTrustBundles can filter on this field
	// using a `spec.signerName=NAME` field selector.
	//
	// +optional
	SignerName string `json:"signerName,omitempty" protobuf:"bytes,1,opt,name=signerName"`

	// trustBundle contains the individual X.509 trust anchors for this
	// bundle, as PEM bundle of PEM-wrapped, DER-formatted X.509 certificates.
	//
	// The data must consist only of PEM certificate blocks that parse as valid
	// X.509 certificates.  Each certificate must include a basic constraints
	// extension with the CA bit set.  The API server will reject objects that
	// contain duplicate certificates, or that use PEM block headers.
	//
	// Users of ClusterTrustBundles, including Kubelet, are free to reorder and
	// deduplicate certificate blocks in this file according to their own logic,
	// as well as to drop PEM block headers and inter-block data.
	TrustBundle string `json:"trustBundle" protobuf:"bytes,2,opt,name=trustBundle"`
}

// +k8s:prerelease-lifecycle-gen:introduced=1.26
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterTrustBundleList is a collection of ClusterTrustBundle objects
type ClusterTrustBundleList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata contains the list metadata.
	//
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a collection of ClusterTrustBundle objects
	Items []ClusterTrustBundle `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// Built-in signerName values that are honored by kube-controller-manager.
const (
	// "kubernetes.io/kube-apiserver-client-pod" issues client certificates that pods can use to authenticate to kube-apiserver.
	// Pods can only obtain these certificates by using PodCertificate projected volumes.
	// Can be auto-approved by the "csrapproving" controller in kube-controller-manager.
	// Can be issued by the "csrsigning" controller in kube-controller-manager.
	KubeAPIServerClientPodSignerName = "kubernetes.io/kube-apiserver-client-pod"
)

// +genclient
// +k8s:prerelease-lifecycle-gen:introduced=1.32
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodCertificateRequest encodes a pod requesting a certificate from a given
// signer.
//
// Kubelets use this API to implement podCertificate projected volumes
type PodCertificateRequest struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec contains the details about the certificate being requested.
	Spec PodCertificateRequestSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status contains the issued certificate, and a standard set of conditions.
	// +optional
	Status PodCertificateRequestStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodCertificateRequestSpec describes the certificate request.  All fields are
// immutable after creation.
type PodCertificateRequestSpec struct {
	// signerName is the requested signer.
	SignerName string `json:"signerName" protobuf:"bytes,1,opt,name=signerName"`

	// podName is the name of the pod into which the certificate will be mounted.
	PodName string `json:"podName" protobuf:"bytes,2,opt,name=podName"`
	// podUID is the UID of the pod into which the certificate will be mounted.
	PodUID types.UID `json:"podUID" protobuf:"bytes,3,opt,name=podUID"`

	// serviceAccountName is the name of the service account the pod is running as.
	ServiceAccountName string `json:"serviceAccountName" protobuf:"bytes,4,opt,name=serviceAccountName"`
	// serviceAccountUID is the UID of the service account the pod is running as.
	ServiceAccountUID types.UID `json:"serviceAccountUID" protobuf:"bytes,5,opt,name=serviceAccountUID"`

	// nodeName is the name of the node the pod is assigned to.
	NodeName types.NodeName `json:"nodeName" protobuf:"bytes,6,opt,name=nodeName"`
	// nodeUID is the UID of the node the pod is assigned to.
	NodeUID types.UID `json:"nodeUID" protobuf:"bytes,7,opt,name=nodeUID"`

	// maxExpirationSeconds is the maximum lifetime permitted for the
	// certificate.
	//
	// If this field is set to 0 during creation of the PodCertificateRequest,
	// then kube-apiserver will set it to 86400(24 hours).  kube-apiserver will
	// reject values shorter than 3600 (1 hour).
	//
	// kube-apiserver will then shorten the value to the maximum expiration
	// configured for the requested signer.
	//
	// The signer implementation is then free to issue a certificate with any
	// lifetime *shorter* than MaxExpirationSeconds.  This constraint is
	// enforced by kube-apiserver.
	MaxExpirationSeconds int32 `json:"expirationSeconds" protobuf:"varint,8,opt,name=expirationSeconds"`

	// pkixPublicKey is the PKIX-serialized public key the signer should issue
	// the certificate to.
	//
	// The key must be one of RSA3072, RSA4096, ECDSAP256, ECDSAP384, or ED25519.
	// Note that this list may be expanded in the future.
	//
	// Signer implementations do not need to support all key types supported by
	// kube-apiserver and kubelet.  If a signer does not support the key type
	// used for a given PodCertificateRequest, it should deny the request, with
	// a reason of UnsupportedKeyType.  It may also suggest a key type that it
	// does support by attaching an additional SuggestedKeyType condition, with
	// its reason field set to the suggested key type identifier.
	PKIXPublicKey []byte `json:"pkixPublicKey" protobuf:"bytes,9,opt,name=pkixPublicKey"`

	// proofOfPossession proves that the requesting kubelet holds the private
	// key corresponding to pkixPublicKey.
	//
	// It is contructed by signing the ASCII bytes of the pod's UID using
	// pkixPublicKey.
	//
	// kube-apiserver validates the proof of possession during creation of the
	// PodCertificateRequest.
	//
	// If the key is an RSA key, then the signature is over the ASCII bytes of
	// the pod UID, using RSASSA-PKCS1-V1_5-SIGN from RSA PKCS #1 v1.5 (as
	// implemented by the golang function crypto/rsa.SignPKCS1v15).
	//
	// If the key is an ECDSA key, then the signature is as described by [SEC 1,
	// Version 2.0](https://www.secg.org/sec1-v2.pdf) (as implemented by the
	// golang library function crypto/ecdsa.SignASN1)
	//
	// If the key is an ED25519 key, the the signature is as described by the
	// [ED25519 Specification](https://ed25519.cr.yp.to/) (as implemented by
	// the golang library crypto/ed25519.Sign).
	ProofOfPossession []byte `json:"proofOfPossession" protobuf:"bytes,10,opt,name=proofOfPossession"`
}

type PodCertificateRequestStatus struct {
	// conditions applied to the request. Known conditions are "Denied",
	// "Failed", and "SuggestedKeyType".
	//
	// If the request is denied with `Reason=UnsupportedKeyType`, the signer
	// may have suggested a key type that will work in the `Reason` field of a
	// `SuggestedKeyType` condition.
	//
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty" protobuf:"bytes,1,rep,name=conditions"`

	// certificateChain is populated with an issued certificate by the signer.
	// This field is set via the /status subresource. Once populated, this field
	// is immutable.
	//
	// If the certificate signing request is denied, a condition of type
	// "Denied" is added and this field remains empty. If the signer cannot
	// issue the certificate, a condition of type "Failed" is added and this
	// field remains empty.
	//
	// Validation requirements:
	//  1. certificateChain must consist of one or more PEM-formatted certificates.
	//  2. Each entry must be a valid PEM-wrapped, DER-encoded ASN.1 Certificate as
	//     described in section 4 of RFC5280.
	//
	// If more than one block is present, and the definition of the requested
	// spec.signerName does not indicate otherwise, the first block is the
	// issued certificate, and subsequent blocks should be treated as
	// intermediate certificates and presented in TLS handshakes.  When
	// projecting the chain into a pod volume, kubelet will preserve the exact
	// contents of certificateChain.
	//
	// +optional
	CertificateChain string `json:"certificateChain,omitempty" protobuf:"bytes,2,opt,name=certificateChain"`

	// issuedAt is the time at which the signer issued the certificate.  This
	// field is set via the /status subresource.  Once populated, it is
	// immutable.  The signer must set this field at the same time it sets
	// certificateChain.
	//
	// +optional
	IssuedAt *metav1.Time `json:"issuedAt,omitempty" protobuf:"bytes,3,opt,name=issuedAt"`

	// notBefore is the time at which the certificate becomes valid.  This field
	// is set via the /status subresource.  Once populated, it is immutable.
	// The signer must set this field at the same time it sets certificateChain.
	//
	// +optional
	NotBefore *metav1.Time `json:"notBefore,omitempty" protobuf:"bytes,4,opt,name=notBefore"`

	// beginRefreshAt is the time at which the kubelet should begin trying to
	// refresh the certificate.  This field is set via the /status subresource,
	// and must be set at the same time as certificateChain.  Once populated,
	// this field is immutable.
	//
	// This field is only a hint.  Kubelet may start refreshing before or after
	// this time if necessary.
	//
	// +optional
	BeginRefreshAt *metav1.Time `json:"beginRefreshAt,omitempty" protobuf:"bytes,5,opt,name=beginRefreshAt"`

	// notAfter is the time at which the certificate expires.  This field is set
	// via the /status subresource.  Once populated, it is immutable.  The
	// signer must set this field at the same time it sets certificateChain.
	//
	// +optional
	NotAfter *metav1.Time `json:"notAfter,omitempty" protobuf:"bytes,6,opt,name=notAfter"`
}

// Well-known condition types for PodCertificateRequests
const (
	// Denied indicates the request was denied by the signer.
	PodCertificateRequestConditionTypeDenied string = "Denied"
	// Failed indicates the signer failed to issue the certificate.
	PodCertificateRequestConditionTypeFailed string = "Failed"
	// SuggestedKeyType is an auxiliary condition that a signer can attach if it
	// denied the request due to an unsupported key type.
	PodCertificateRequestConditionTypeSuggestedKeyType string = "SuggestedKeyType"
)

// Well-known condition reasons for PodCertificateRequests
const (
	// UnsupportedKeyType should be set on "Denied" conditions when the signer
	// doesn't support the key type of publicKey.
	PodCertificateRequestConditionUnsupportedKeyType string = "UnsupportedKeyType"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.32

// PodCertificateRequestList is a collection of PodCertificateRequest objects
type PodCertificateRequestList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a collection of PodCertificateRequest objects
	Items []PodCertificateRequest `json:"items" protobuf:"bytes,2,rep,name=items"`
}
