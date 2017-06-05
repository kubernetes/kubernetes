package v2

import (
	"net/http"

	"github.com/docker/distribution/registry/api/errcode"
)

const errGroup = "registry.api.v2"

var (
	// ErrorCodeDigestInvalid is returned when uploading a blob if the
	// provided digest does not match the blob contents.
	ErrorCodeDigestInvalid = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "DIGEST_INVALID",
		Message: "provided digest did not match uploaded content",
		Description: `When a blob is uploaded, the registry will check that
		the content matches the digest provided by the client. The error may
		include a detail structure with the key "digest", including the
		invalid digest string. This error may also be returned when a manifest
		includes an invalid layer digest.`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorCodeSizeInvalid is returned when uploading a blob if the provided
	ErrorCodeSizeInvalid = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "SIZE_INVALID",
		Message: "provided length did not match content length",
		Description: `When a layer is uploaded, the provided size will be
		checked against the uploaded content. If they do not match, this error
		will be returned.`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorCodeNameInvalid is returned when the name in the manifest does not
	// match the provided name.
	ErrorCodeNameInvalid = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "NAME_INVALID",
		Message: "invalid repository name",
		Description: `Invalid repository name encountered either during
		manifest validation or any API operation.`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorCodeTagInvalid is returned when the tag in the manifest does not
	// match the provided tag.
	ErrorCodeTagInvalid = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "TAG_INVALID",
		Message: "manifest tag did not match URI",
		Description: `During a manifest upload, if the tag in the manifest
		does not match the uri tag, this error will be returned.`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorCodeNameUnknown when the repository name is not known.
	ErrorCodeNameUnknown = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "NAME_UNKNOWN",
		Message: "repository name not known to registry",
		Description: `This is returned if the name used during an operation is
		unknown to the registry.`,
		HTTPStatusCode: http.StatusNotFound,
	})

	// ErrorCodeManifestUnknown returned when image manifest is unknown.
	ErrorCodeManifestUnknown = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "MANIFEST_UNKNOWN",
		Message: "manifest unknown",
		Description: `This error is returned when the manifest, identified by
		name and tag is unknown to the repository.`,
		HTTPStatusCode: http.StatusNotFound,
	})

	// ErrorCodeManifestInvalid returned when an image manifest is invalid,
	// typically during a PUT operation. This error encompasses all errors
	// encountered during manifest validation that aren't signature errors.
	ErrorCodeManifestInvalid = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "MANIFEST_INVALID",
		Message: "manifest invalid",
		Description: `During upload, manifests undergo several checks ensuring
		validity. If those checks fail, this error may be returned, unless a
		more specific error is included. The detail will contain information
		the failed validation.`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorCodeManifestUnverified is returned when the manifest fails
	// signature verification.
	ErrorCodeManifestUnverified = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "MANIFEST_UNVERIFIED",
		Message: "manifest failed signature verification",
		Description: `During manifest upload, if the manifest fails signature
		verification, this error will be returned.`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorCodeManifestBlobUnknown is returned when a manifest blob is
	// unknown to the registry.
	ErrorCodeManifestBlobUnknown = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "MANIFEST_BLOB_UNKNOWN",
		Message: "blob unknown to registry",
		Description: `This error may be returned when a manifest blob is 
		unknown to the registry.`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorCodeBlobUnknown is returned when a blob is unknown to the
	// registry. This can happen when the manifest references a nonexistent
	// layer or the result is not found by a blob fetch.
	ErrorCodeBlobUnknown = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "BLOB_UNKNOWN",
		Message: "blob unknown to registry",
		Description: `This error may be returned when a blob is unknown to the
		registry in a specified repository. This can be returned with a
		standard get or if a manifest references an unknown layer during
		upload.`,
		HTTPStatusCode: http.StatusNotFound,
	})

	// ErrorCodeBlobUploadUnknown is returned when an upload is unknown.
	ErrorCodeBlobUploadUnknown = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "BLOB_UPLOAD_UNKNOWN",
		Message: "blob upload unknown to registry",
		Description: `If a blob upload has been cancelled or was never
		started, this error code may be returned.`,
		HTTPStatusCode: http.StatusNotFound,
	})

	// ErrorCodeBlobUploadInvalid is returned when an upload is invalid.
	ErrorCodeBlobUploadInvalid = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "BLOB_UPLOAD_INVALID",
		Message: "blob upload invalid",
		Description: `The blob upload encountered an error and can no
		longer proceed.`,
		HTTPStatusCode: http.StatusNotFound,
	})
)
