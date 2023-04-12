package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageList is a list of Image objects.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of images
	Items []Image `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Image is an immutable representation of a container image and metadata at a point in time.
// Images are named by taking a hash of their contents (metadata and content) and any change
// in format, content, or metadata results in a new name. The images resource is primarily
// for use by cluster administrators and integrations like the cluster image registry - end
// users instead access images via the imagestreamtags or imagestreamimages resources. While
// image metadata is stored in the API, any integration that implements the container image
// registry API must provide its own storage for the raw manifest data, image config, and
// layer contents.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Image struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// DockerImageReference is the string that can be used to pull this image.
	DockerImageReference string `json:"dockerImageReference,omitempty" protobuf:"bytes,2,opt,name=dockerImageReference"`
	// DockerImageMetadata contains metadata about this image
	// +patchStrategy=replace
	// +kubebuilder:pruning:PreserveUnknownFields
	DockerImageMetadata runtime.RawExtension `json:"dockerImageMetadata,omitempty" patchStrategy:"replace" protobuf:"bytes,3,opt,name=dockerImageMetadata"`
	// DockerImageMetadataVersion conveys the version of the object, which if empty defaults to "1.0"
	DockerImageMetadataVersion string `json:"dockerImageMetadataVersion,omitempty" protobuf:"bytes,4,opt,name=dockerImageMetadataVersion"`
	// DockerImageManifest is the raw JSON of the manifest
	DockerImageManifest string `json:"dockerImageManifest,omitempty" protobuf:"bytes,5,opt,name=dockerImageManifest"`
	// DockerImageLayers represents the layers in the image. May not be set if the image does not define that data or if the image represents a manifest list.
	DockerImageLayers []ImageLayer `json:"dockerImageLayers,omitempty" protobuf:"bytes,6,rep,name=dockerImageLayers"`
	// Signatures holds all signatures of the image.
	// +patchMergeKey=name
	// +patchStrategy=merge
	Signatures []ImageSignature `json:"signatures,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,7,rep,name=signatures"`
	// DockerImageSignatures provides the signatures as opaque blobs. This is a part of manifest schema v1.
	DockerImageSignatures [][]byte `json:"dockerImageSignatures,omitempty" protobuf:"bytes,8,rep,name=dockerImageSignatures"`
	// DockerImageManifestMediaType specifies the mediaType of manifest. This is a part of manifest schema v2.
	DockerImageManifestMediaType string `json:"dockerImageManifestMediaType,omitempty" protobuf:"bytes,9,opt,name=dockerImageManifestMediaType"`
	// DockerImageConfig is a JSON blob that the runtime uses to set up the container. This is a part of manifest schema v2.
	// Will not be set when the image represents a manifest list.
	DockerImageConfig string `json:"dockerImageConfig,omitempty" protobuf:"bytes,10,opt,name=dockerImageConfig"`
	// DockerImageManifests holds information about sub-manifests when the image represents a manifest list.
	// When this field is present, no DockerImageLayers should be specified.
	DockerImageManifests []ImageManifest `json:"dockerImageManifests,omitempty" protobuf:"bytes,11,rep,name=dockerImageManifests"`
}

// ImageManifest represents sub-manifests of a manifest list. The Digest field points to a regular
// Image object.
type ImageManifest struct {
	// Digest is the unique identifier for the manifest. It refers to an Image object.
	Digest string `json:"digest" protobuf:"bytes,1,opt,name=digest"`
	// MediaType defines the type of the manifest, possible values are application/vnd.oci.image.manifest.v1+json,
	// application/vnd.docker.distribution.manifest.v2+json or application/vnd.docker.distribution.manifest.v1+json.
	MediaType string `json:"mediaType" protobuf:"bytes,2,opt,name=mediaType"`
	// ManifestSize represents the size of the raw object contents, in bytes.
	ManifestSize int64 `json:"manifestSize" protobuf:"varint,3,opt,name=manifestSize"`
	// Architecture specifies the supported CPU architecture, for example `amd64` or `ppc64le`.
	Architecture string `json:"architecture" protobuf:"bytes,4,opt,name=architecture"`
	// OS specifies the operating system, for example `linux`.
	OS string `json:"os" protobuf:"bytes,5,opt,name=os"`
	// Variant is an optional field repreenting a variant of the CPU, for example v6 to specify a particular CPU
	// variant of the ARM CPU.
	Variant string `json:"variant,omitempty" protobuf:"bytes,6,opt,name=variant"`
}

// ImageLayer represents a single layer of the image. Some images may have multiple layers. Some may have none.
type ImageLayer struct {
	// Name of the layer as defined by the underlying store.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// Size of the layer in bytes as defined by the underlying store.
	LayerSize int64 `json:"size" protobuf:"varint,2,opt,name=size"`
	// MediaType of the referenced object.
	MediaType string `json:"mediaType" protobuf:"bytes,3,opt,name=mediaType"`
}

// +genclient
// +genclient:onlyVerbs=create,delete
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageSignature holds a signature of an image. It allows to verify image identity and possibly other claims
// as long as the signature is trusted. Based on this information it is possible to restrict runnable images
// to those matching cluster-wide policy.
// Mandatory fields should be parsed by clients doing image verification. The others are parsed from
// signature's content by the server. They serve just an informative purpose.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageSignature struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Required: Describes a type of stored blob.
	Type string `json:"type" protobuf:"bytes,2,opt,name=type"`
	// Required: An opaque binary string which is an image's signature.
	Content []byte `json:"content" protobuf:"bytes,3,opt,name=content"`
	// Conditions represent the latest available observations of a signature's current state.
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []SignatureCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,4,rep,name=conditions"`

	// Following metadata fields will be set by server if the signature content is successfully parsed and
	// the information available.

	// A human readable string representing image's identity. It could be a product name and version, or an
	// image pull spec (e.g. "registry.access.redhat.com/rhel7/rhel:7.2").
	ImageIdentity string `json:"imageIdentity,omitempty" protobuf:"bytes,5,opt,name=imageIdentity"`
	// Contains claims from the signature.
	SignedClaims map[string]string `json:"signedClaims,omitempty" protobuf:"bytes,6,rep,name=signedClaims"`
	// If specified, it is the time of signature's creation.
	Created *metav1.Time `json:"created,omitempty" protobuf:"bytes,7,opt,name=created"`
	// If specified, it holds information about an issuer of signing certificate or key (a person or entity
	// who signed the signing certificate or key).
	IssuedBy *SignatureIssuer `json:"issuedBy,omitempty" protobuf:"bytes,8,opt,name=issuedBy"`
	// If specified, it holds information about a subject of signing certificate or key (a person or entity
	// who signed the image).
	IssuedTo *SignatureSubject `json:"issuedTo,omitempty" protobuf:"bytes,9,opt,name=issuedTo"`
}

// SignatureConditionType is a type of image signature condition.
type SignatureConditionType string

// SignatureCondition describes an image signature condition of particular kind at particular probe time.
type SignatureCondition struct {
	// Type of signature condition, Complete or Failed.
	Type SignatureConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=SignatureConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status corev1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/api/core/v1.ConditionStatus"`
	// Last time the condition was checked.
	LastProbeTime metav1.Time `json:"lastProbeTime,omitempty" protobuf:"bytes,3,opt,name=lastProbeTime"`
	// Last time the condition transit from one status to another.
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,4,opt,name=lastTransitionTime"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty" protobuf:"bytes,5,opt,name=reason"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty" protobuf:"bytes,6,opt,name=message"`
}

// SignatureGenericEntity holds a generic information about a person or entity who is an issuer or a subject
// of signing certificate or key.
type SignatureGenericEntity struct {
	// Organization name.
	Organization string `json:"organization,omitempty" protobuf:"bytes,1,opt,name=organization"`
	// Common name (e.g. openshift-signing-service).
	CommonName string `json:"commonName,omitempty" protobuf:"bytes,2,opt,name=commonName"`
}

// SignatureIssuer holds information about an issuer of signing certificate or key.
type SignatureIssuer struct {
	SignatureGenericEntity `json:",inline" protobuf:"bytes,1,opt,name=signatureGenericEntity"`
}

// SignatureSubject holds information about a person or entity who created the signature.
type SignatureSubject struct {
	SignatureGenericEntity `json:",inline" protobuf:"bytes,1,opt,name=signatureGenericEntity"`
	// If present, it is a human readable key id of public key belonging to the subject used to verify image
	// signature. It should contain at least 64 lowest bits of public key's fingerprint (e.g.
	// 0x685ebe62bf278440).
	PublicKeyID string `json:"publicKeyID" protobuf:"bytes,2,opt,name=publicKeyID"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageStreamList is a list of ImageStream objects.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStreamList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of imageStreams
	Items []ImageStream `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:method=Secrets,verb=get,subresource=secrets,result=github.com/openshift/api/image/v1.SecretList
// +genclient:method=Layers,verb=get,subresource=layers,result=github.com/openshift/api/image/v1.ImageStreamLayers
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// An ImageStream stores a mapping of tags to images, metadata overrides that are applied
// when images are tagged in a stream, and an optional reference to a container image
// repository on a registry. Users typically update the spec.tags field to point to external
// images which are imported from container registries using credentials in your namespace
// with the pull secret type, or to existing image stream tags and images which are
// immediately accessible for tagging or pulling. The history of images applied to a tag
// is visible in the status.tags field and any user who can view an image stream is allowed
// to tag that image into their own image streams. Access to pull images from the integrated
// registry is granted by having the "get imagestreams/layers" permission on a given image
// stream. Users may remove a tag by deleting the imagestreamtag resource, which causes both
// spec and status for that tag to be removed. Image stream history is retained until an
// administrator runs the prune operation, which removes references that are no longer in
// use. To preserve a historical image, ensure there is a tag in spec pointing to that image
// by its digest.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStream struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec describes the desired state of this stream
	// +optional
	Spec ImageStreamSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
	// Status describes the current state of this stream
	// +optional
	Status ImageStreamStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ImageStreamSpec represents options for ImageStreams.
type ImageStreamSpec struct {
	// lookupPolicy controls how other resources reference images within this namespace.
	LookupPolicy ImageLookupPolicy `json:"lookupPolicy,omitempty" protobuf:"bytes,3,opt,name=lookupPolicy"`
	// dockerImageRepository is optional, if specified this stream is backed by a container repository on this server
	// Deprecated: This field is deprecated as of v3.7 and will be removed in a future release.
	// Specify the source for the tags to be imported in each tag via the spec.tags.from reference instead.
	DockerImageRepository string `json:"dockerImageRepository,omitempty" protobuf:"bytes,1,opt,name=dockerImageRepository"`
	// tags map arbitrary string values to specific image locators
	// +patchMergeKey=name
	// +patchStrategy=merge
	Tags []TagReference `json:"tags,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,2,rep,name=tags"`
}

// ImageLookupPolicy describes how an image stream can be used to override the image references
// used by pods, builds, and other resources in a namespace.
type ImageLookupPolicy struct {
	// local will change the docker short image references (like "mysql" or
	// "php:latest") on objects in this namespace to the image ID whenever they match
	// this image stream, instead of reaching out to a remote registry. The name will
	// be fully qualified to an image ID if found. The tag's referencePolicy is taken
	// into account on the replaced value. Only works within the current namespace.
	Local bool `json:"local" protobuf:"varint,3,opt,name=local"`
}

// TagReference specifies optional annotations for images using this tag and an optional reference to an ImageStreamTag, ImageStreamImage, or DockerImage this tag should track.
type TagReference struct {
	// Name of the tag
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// Optional; if specified, annotations that are applied to images retrieved via ImageStreamTags.
	// +optional
	Annotations map[string]string `json:"annotations" protobuf:"bytes,2,rep,name=annotations"`
	// Optional; if specified, a reference to another image that this tag should point to. Valid values
	// are ImageStreamTag, ImageStreamImage, and DockerImage.  ImageStreamTag references
	// can only reference a tag within this same ImageStream.
	From *corev1.ObjectReference `json:"from,omitempty" protobuf:"bytes,3,opt,name=from"`
	// Reference states if the tag will be imported. Default value is false, which means the tag will
	// be imported.
	Reference bool `json:"reference,omitempty" protobuf:"varint,4,opt,name=reference"`
	// Generation is a counter that tracks mutations to the spec tag (user intent). When a tag reference
	// is changed the generation is set to match the current stream generation (which is incremented every
	// time spec is changed). Other processes in the system like the image importer observe that the
	// generation of spec tag is newer than the generation recorded in the status and use that as a trigger
	// to import the newest remote tag. To trigger a new import, clients may set this value to zero which
	// will reset the generation to the latest stream generation. Legacy clients will send this value as
	// nil which will be merged with the current tag generation.
	// +optional
	Generation *int64 `json:"generation" protobuf:"varint,5,opt,name=generation"`
	// ImportPolicy is information that controls how images may be imported by the server.
	ImportPolicy TagImportPolicy `json:"importPolicy,omitempty" protobuf:"bytes,6,opt,name=importPolicy"`
	// ReferencePolicy defines how other components should consume the image.
	ReferencePolicy TagReferencePolicy `json:"referencePolicy,omitempty" protobuf:"bytes,7,opt,name=referencePolicy"`
}

// TagImportPolicy controls how images related to this tag will be imported.
type TagImportPolicy struct {
	// Insecure is true if the server may bypass certificate verification or connect directly over HTTP during image import.
	Insecure bool `json:"insecure,omitempty" protobuf:"varint,1,opt,name=insecure"`
	// Scheduled indicates to the server that this tag should be periodically checked to ensure it is up to date, and imported
	Scheduled bool `json:"scheduled,omitempty" protobuf:"varint,2,opt,name=scheduled"`
	// ImportMode describes how to import an image manifest.
	ImportMode ImportModeType `json:"importMode,omitempty" protobuf:"bytes,3,opt,name=importMode,casttype=ImportModeType"`
}

// ImportModeType describes how to import an image manifest.
type ImportModeType string

const (
	// ImportModeLegacy indicates that the legacy behaviour should be used.
	// For manifest lists, the legacy behaviour will discard the manifest list and import a single
	// sub-manifest. In this case, the platform is chosen in the following order of priority:
	// 1. tag annotations; 2. control plane arch/os; 3. linux/amd64; 4. the first manifest in the list.
	// This mode is the default.
	ImportModeLegacy ImportModeType = "Legacy"
	// ImportModePreserveOriginal indicates that the original manifest will be preserved.
	// For manifest lists, the manifest list and all its sub-manifests will be imported.
	ImportModePreserveOriginal ImportModeType = "PreserveOriginal"
)

// TagReferencePolicyType describes how pull-specs for images in an image stream tag are generated when
// image change triggers are fired.
type TagReferencePolicyType string

const (
	// SourceTagReferencePolicy indicates the image's original location should be used when the image stream tag
	// is resolved into other resources (builds and deployment configurations).
	SourceTagReferencePolicy TagReferencePolicyType = "Source"
	// LocalTagReferencePolicy indicates the image should prefer to pull via the local integrated registry,
	// falling back to the remote location if the integrated registry has not been configured. The reference will
	// use the internal DNS name or registry service IP.
	LocalTagReferencePolicy TagReferencePolicyType = "Local"
)

// TagReferencePolicy describes how pull-specs for images in this image stream tag are generated when
// image change triggers in deployment configs or builds are resolved. This allows the image stream
// author to control how images are accessed.
type TagReferencePolicy struct {
	// Type determines how the image pull spec should be transformed when the image stream tag is used in
	// deployment config triggers or new builds. The default value is `Source`, indicating the original
	// location of the image should be used (if imported). The user may also specify `Local`, indicating
	// that the pull spec should point to the integrated container image registry and leverage the registry's
	// ability to proxy the pull to an upstream registry. `Local` allows the credentials used to pull this
	// image to be managed from the image stream's namespace, so others on the platform can access a remote
	// image but have no access to the remote secret. It also allows the image layers to be mirrored into
	// the local registry which the images can still be pulled even if the upstream registry is unavailable.
	Type TagReferencePolicyType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=TagReferencePolicyType"`
}

// ImageStreamStatus contains information about the state of this image stream.
type ImageStreamStatus struct {
	// DockerImageRepository represents the effective location this stream may be accessed at.
	// May be empty until the server determines where the repository is located
	DockerImageRepository string `json:"dockerImageRepository" protobuf:"bytes,1,opt,name=dockerImageRepository"`
	// PublicDockerImageRepository represents the public location from where the image can
	// be pulled outside the cluster. This field may be empty if the administrator
	// has not exposed the integrated registry externally.
	PublicDockerImageRepository string `json:"publicDockerImageRepository,omitempty" protobuf:"bytes,3,opt,name=publicDockerImageRepository"`
	// Tags are a historical record of images associated with each tag. The first entry in the
	// TagEvent array is the currently tagged image.
	// +patchMergeKey=tag
	// +patchStrategy=merge
	Tags []NamedTagEventList `json:"tags,omitempty" patchStrategy:"merge" patchMergeKey:"tag" protobuf:"bytes,2,rep,name=tags"`
}

// NamedTagEventList relates a tag to its image history.
type NamedTagEventList struct {
	// Tag is the tag for which the history is recorded
	Tag string `json:"tag" protobuf:"bytes,1,opt,name=tag"`
	// Standard object's metadata.
	Items []TagEvent `json:"items" protobuf:"bytes,2,rep,name=items"`
	// Conditions is an array of conditions that apply to the tag event list.
	Conditions []TagEventCondition `json:"conditions,omitempty" protobuf:"bytes,3,rep,name=conditions"`
}

// TagEvent is used by ImageStreamStatus to keep a historical record of images associated with a tag.
type TagEvent struct {
	// Created holds the time the TagEvent was created
	Created metav1.Time `json:"created" protobuf:"bytes,1,opt,name=created"`
	// DockerImageReference is the string that can be used to pull this image
	DockerImageReference string `json:"dockerImageReference" protobuf:"bytes,2,opt,name=dockerImageReference"`
	// Image is the image
	Image string `json:"image" protobuf:"bytes,3,opt,name=image"`
	// Generation is the spec tag generation that resulted in this tag being updated
	Generation int64 `json:"generation" protobuf:"varint,4,opt,name=generation"`
}

type TagEventConditionType string

// These are valid conditions of TagEvents.
const (
	// ImportSuccess with status False means the import of the specific tag failed
	ImportSuccess TagEventConditionType = "ImportSuccess"
)

// TagEventCondition contains condition information for a tag event.
type TagEventCondition struct {
	// Type of tag event condition, currently only ImportSuccess
	Type TagEventConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=TagEventConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status corev1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/api/core/v1.ConditionStatus"`
	// LastTransitionTIme is the time the condition transitioned from one status to another.
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,3,opt,name=lastTransitionTime"`
	// Reason is a brief machine readable explanation for the condition's last transition.
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`
	// Message is a human readable description of the details about last transition, complementing reason.
	Message string `json:"message,omitempty" protobuf:"bytes,5,opt,name=message"`
	// Generation is the spec tag generation that this status corresponds to
	Generation int64 `json:"generation" protobuf:"varint,6,opt,name=generation"`
}

// +genclient
// +genclient:skipVerbs=get,list,create,update,patch,delete,deleteCollection,watch
// +genclient:method=Create,verb=create,result=k8s.io/apimachinery/pkg/apis/meta/v1.Status
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageStreamMapping represents a mapping from a single image stream tag to a container
// image as well as the reference to the container image stream the image came from. This
// resource is used by privileged integrators to create an image resource and to associate
// it with an image stream in the status tags field. Creating an ImageStreamMapping will
// allow any user who can view the image stream to tag or pull that image, so only create
// mappings where the user has proven they have access to the image contents directly.
// The only operation supported for this resource is create and the metadata name and
// namespace should be set to the image stream containing the tag that should be updated.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStreamMapping struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Image is a container image.
	Image Image `json:"image" protobuf:"bytes,2,opt,name=image"`
	// Tag is a string value this image can be located with inside the stream.
	Tag string `json:"tag" protobuf:"bytes,3,opt,name=tag"`
}

// +genclient
// +genclient:onlyVerbs=get,list,create,update,delete
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageStreamTag represents an Image that is retrieved by tag name from an ImageStream.
// Use this resource to interact with the tags and images in an image stream by tag, or
// to see the image details for a particular tag. The image associated with this resource
// is the most recently successfully tagged, imported, or pushed image (as described in the
// image stream status.tags.items list for this tag). If an import is in progress or has
// failed the previous image will be shown. Deleting an image stream tag clears both the
// status and spec fields of an image stream. If no image can be retrieved for a given tag,
// a not found error will be returned.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStreamTag struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// tag is the spec tag associated with this image stream tag, and it may be null
	// if only pushes have occurred to this image stream.
	Tag *TagReference `json:"tag" protobuf:"bytes,2,opt,name=tag"`

	// generation is the current generation of the tagged image - if tag is provided
	// and this value is not equal to the tag generation, a user has requested an
	// import that has not completed, or conditions will be filled out indicating any
	// error.
	Generation int64 `json:"generation" protobuf:"varint,3,opt,name=generation"`

	// lookupPolicy indicates whether this tag will handle image references in this
	// namespace.
	LookupPolicy ImageLookupPolicy `json:"lookupPolicy" protobuf:"varint,6,opt,name=lookupPolicy"`

	// conditions is an array of conditions that apply to the image stream tag.
	Conditions []TagEventCondition `json:"conditions,omitempty" protobuf:"bytes,4,rep,name=conditions"`

	// image associated with the ImageStream and tag.
	Image Image `json:"image" protobuf:"bytes,5,opt,name=image"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageStreamTagList is a list of ImageStreamTag objects.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStreamTagList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of image stream tags
	Items []ImageStreamTag `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:onlyVerbs=get,list,create,update,delete
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageTag represents a single tag within an image stream and includes the spec,
// the status history, and the currently referenced image (if any) of the provided
// tag. This type replaces the ImageStreamTag by providing a full view of the tag.
// ImageTags are returned for every spec or status tag present on the image stream.
// If no tag exists in either form a not found error will be returned by the API.
// A create operation will succeed if no spec tag has already been defined and the
// spec field is set. Delete will remove both spec and status elements from the
// image stream.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageTag struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec is the spec tag associated with this image stream tag, and it may be null
	// if only pushes have occurred to this image stream.
	Spec *TagReference `json:"spec" protobuf:"bytes,2,opt,name=spec"`
	// status is the status tag details associated with this image stream tag, and it
	// may be null if no push or import has been performed.
	Status *NamedTagEventList `json:"status" protobuf:"bytes,3,opt,name=status"`
	// image is the details of the most recent image stream status tag, and it may be
	// null if import has not completed or an administrator has deleted the image
	// object. To verify this is the most recent image, you must verify the generation
	// of the most recent status.items entry matches the spec tag (if a spec tag is
	// set). This field will not be set when listing image tags.
	Image *Image `json:"image" protobuf:"bytes,4,opt,name=image"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageTagList is a list of ImageTag objects. When listing image tags, the image
// field is not populated. Tags are returned in alphabetical order by image stream
// and then tag.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageTagList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of image stream tags
	Items []ImageTag `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:onlyVerbs=get
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageStreamImage represents an Image that is retrieved by image name from an ImageStream.
// User interfaces and regular users can use this resource to access the metadata details of
// a tagged image in the image stream history for viewing, since Image resources are not
// directly accessible to end users. A not found error will be returned if no such image is
// referenced by a tag within the ImageStream. Images are created when spec tags are set on
// an image stream that represent an image in an external registry, when pushing to the
// integrated registry, or when tagging an existing image from one image stream to another.
// The name of an image stream image is in the form "<STREAM>@<DIGEST>", where the digest is
// the content addressible identifier for the image (sha256:xxxxx...). You can use
// ImageStreamImages as the from.kind of an image stream spec tag to reference an image
// exactly. The only operations supported on the imagestreamimage endpoint are retrieving
// the image.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStreamImage struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Image associated with the ImageStream and image name.
	Image Image `json:"image" protobuf:"bytes,2,opt,name=image"`
}

// DockerImageReference points to a container image.
type DockerImageReference struct {
	// Registry is the registry that contains the container image
	Registry string `protobuf:"bytes,1,opt,name=registry"`
	// Namespace is the namespace that contains the container image
	Namespace string `protobuf:"bytes,2,opt,name=namespace"`
	// Name is the name of the container image
	Name string `protobuf:"bytes,3,opt,name=name"`
	// Tag is which tag of the container image is being referenced
	Tag string `protobuf:"bytes,4,opt,name=tag"`
	// ID is the identifier for the container image
	ID string `protobuf:"bytes,5,opt,name=iD"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageStreamLayers describes information about the layers referenced by images in this
// image stream.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStreamLayers struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// blobs is a map of blob name to metadata about the blob.
	Blobs map[string]ImageLayerData `json:"blobs" protobuf:"bytes,2,rep,name=blobs"`
	// images is a map between an image name and the names of the blobs and config that
	// comprise the image.
	Images map[string]ImageBlobReferences `json:"images" protobuf:"bytes,3,rep,name=images"`
}

// ImageBlobReferences describes the blob references within an image.
type ImageBlobReferences struct {
	// imageMissing is true if the image is referenced by the image stream but the image
	// object has been deleted from the API by an administrator. When this field is set,
	// layers and config fields may be empty and callers that depend on the image metadata
	// should consider the image to be unavailable for download or viewing.
	// +optional
	ImageMissing bool `json:"imageMissing" protobuf:"varint,3,opt,name=imageMissing"`
	// layers is the list of blobs that compose this image, from base layer to top layer.
	// All layers referenced by this array will be defined in the blobs map. Some images
	// may have zero layers.
	// +optional
	Layers []string `json:"layers" protobuf:"bytes,1,rep,name=layers"`
	// config, if set, is the blob that contains the image config. Some images do
	// not have separate config blobs and this field will be set to nil if so.
	// +optional
	Config *string `json:"config" protobuf:"bytes,2,opt,name=config"`
	// manifests is the list of other image names that this image points
	// to. For a single architecture image, it is empty. For a multi-arch
	// image, it consists of the digests of single architecture images,
	// such images shouldn't have layers nor config.
	// +optional
	Manifests []string `json:"manifests,omitempty" protobuf:"bytes,4,rep,name=manifests"`
}

// ImageLayerData contains metadata about an image layer.
type ImageLayerData struct {
	// Size of the layer in bytes as defined by the underlying store. This field is
	// optional if the necessary information about size is not available.
	LayerSize *int64 `json:"size" protobuf:"varint,1,opt,name=size"`
	// MediaType of the referenced object.
	MediaType string `json:"mediaType" protobuf:"bytes,2,opt,name=mediaType"`
}

// +genclient
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// The image stream import resource provides an easy way for a user to find and import container images
// from other container image registries into the server. Individual images or an entire image repository may
// be imported, and users may choose to see the results of the import prior to tagging the resulting
// images into the specified image stream.
//
// This API is intended for end-user tools that need to see the metadata of the image prior to import
// (for instance, to generate an application from it). Clients that know the desired image can continue
// to create spec.tags directly into their image streams.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageStreamImport struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec is a description of the images that the user wishes to import
	Spec ImageStreamImportSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
	// Status is the result of importing the image
	Status ImageStreamImportStatus `json:"status" protobuf:"bytes,3,opt,name=status"`
}

// ImageStreamImportSpec defines what images should be imported.
type ImageStreamImportSpec struct {
	// Import indicates whether to perform an import - if so, the specified tags are set on the spec
	// and status of the image stream defined by the type meta.
	Import bool `json:"import" protobuf:"varint,1,opt,name=import"`
	// Repository is an optional import of an entire container image repository. A maximum limit on the
	// number of tags imported this way is imposed by the server.
	Repository *RepositoryImportSpec `json:"repository,omitempty" protobuf:"bytes,2,opt,name=repository"`
	// Images are a list of individual images to import.
	Images []ImageImportSpec `json:"images,omitempty" protobuf:"bytes,3,rep,name=images"`
}

// ImageStreamImportStatus contains information about the status of an image stream import.
type ImageStreamImportStatus struct {
	// Import is the image stream that was successfully updated or created when 'to' was set.
	Import *ImageStream `json:"import,omitempty" protobuf:"bytes,1,opt,name=import"`
	// Repository is set if spec.repository was set to the outcome of the import
	Repository *RepositoryImportStatus `json:"repository,omitempty" protobuf:"bytes,2,opt,name=repository"`
	// Images is set with the result of importing spec.images
	Images []ImageImportStatus `json:"images,omitempty" protobuf:"bytes,3,rep,name=images"`
}

// RepositoryImportSpec describes a request to import images from a container image repository.
type RepositoryImportSpec struct {
	// From is the source for the image repository to import; only kind DockerImage and a name of a container image repository is allowed
	From corev1.ObjectReference `json:"from" protobuf:"bytes,1,opt,name=from"`

	// ImportPolicy is the policy controlling how the image is imported
	ImportPolicy TagImportPolicy `json:"importPolicy,omitempty" protobuf:"bytes,2,opt,name=importPolicy"`
	// ReferencePolicy defines how other components should consume the image
	ReferencePolicy TagReferencePolicy `json:"referencePolicy,omitempty" protobuf:"bytes,4,opt,name=referencePolicy"`
	// IncludeManifest determines if the manifest for each image is returned in the response
	IncludeManifest bool `json:"includeManifest,omitempty" protobuf:"varint,3,opt,name=includeManifest"`
}

// RepositoryImportStatus describes the result of an image repository import
type RepositoryImportStatus struct {
	// Status reflects whether any failure occurred during import
	Status metav1.Status `json:"status,omitempty" protobuf:"bytes,1,opt,name=status"`
	// Images is a list of images successfully retrieved by the import of the repository.
	Images []ImageImportStatus `json:"images,omitempty" protobuf:"bytes,2,rep,name=images"`
	// AdditionalTags are tags that exist in the repository but were not imported because
	// a maximum limit of automatic imports was applied.
	AdditionalTags []string `json:"additionalTags,omitempty" protobuf:"bytes,3,rep,name=additionalTags"`
}

// ImageImportSpec describes a request to import a specific image.
type ImageImportSpec struct {
	// From is the source of an image to import; only kind DockerImage is allowed
	From corev1.ObjectReference `json:"from" protobuf:"bytes,1,opt,name=from"`
	// To is a tag in the current image stream to assign the imported image to, if name is not specified the default tag from from.name will be used
	To *corev1.LocalObjectReference `json:"to,omitempty" protobuf:"bytes,2,opt,name=to"`

	// ImportPolicy is the policy controlling how the image is imported
	ImportPolicy TagImportPolicy `json:"importPolicy,omitempty" protobuf:"bytes,3,opt,name=importPolicy"`
	// ReferencePolicy defines how other components should consume the image
	ReferencePolicy TagReferencePolicy `json:"referencePolicy,omitempty" protobuf:"bytes,5,opt,name=referencePolicy"`
	// IncludeManifest determines if the manifest for each image is returned in the response
	IncludeManifest bool `json:"includeManifest,omitempty" protobuf:"varint,4,opt,name=includeManifest"`
}

// ImageImportStatus describes the result of an image import.
type ImageImportStatus struct {
	// Status is the status of the image import, including errors encountered while retrieving the image
	Status metav1.Status `json:"status" protobuf:"bytes,1,opt,name=status"`
	// Image is the metadata of that image, if the image was located
	Image *Image `json:"image,omitempty" protobuf:"bytes,2,opt,name=image"`
	// Tag is the tag this image was located under, if any
	Tag string `json:"tag,omitempty" protobuf:"bytes,3,opt,name=tag"`
	// Manifests holds sub-manifests metadata when importing a manifest list
	Manifests []Image `json:"manifests,omitempty" protobuf:"bytes,4,rep,name=manifests"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SecretList is a list of Secret.
// +openshift:compatibility-gen:level=1
type SecretList corev1.SecretList
