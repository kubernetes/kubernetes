package distribution

import (
	"errors"
	"fmt"
	"strings"

	"github.com/docker/distribution/digest"
)

// ErrAccessDenied is returned when an access to a requested resource is
// denied.
var ErrAccessDenied = errors.New("access denied")

// ErrManifestNotModified is returned when a conditional manifest GetByTag
// returns nil due to the client indicating it has the latest version
var ErrManifestNotModified = errors.New("manifest not modified")

// ErrUnsupported is returned when an unimplemented or unsupported action is
// performed
var ErrUnsupported = errors.New("operation unsupported")

// ErrTagUnknown is returned if the given tag is not known by the tag service
type ErrTagUnknown struct {
	Tag string
}

func (err ErrTagUnknown) Error() string {
	return fmt.Sprintf("unknown tag=%s", err.Tag)
}

// ErrRepositoryUnknown is returned if the named repository is not known by
// the registry.
type ErrRepositoryUnknown struct {
	Name string
}

func (err ErrRepositoryUnknown) Error() string {
	return fmt.Sprintf("unknown repository name=%s", err.Name)
}

// ErrRepositoryNameInvalid should be used to denote an invalid repository
// name. Reason may set, indicating the cause of invalidity.
type ErrRepositoryNameInvalid struct {
	Name   string
	Reason error
}

func (err ErrRepositoryNameInvalid) Error() string {
	return fmt.Sprintf("repository name %q invalid: %v", err.Name, err.Reason)
}

// ErrManifestUnknown is returned if the manifest is not known by the
// registry.
type ErrManifestUnknown struct {
	Name string
	Tag  string
}

func (err ErrManifestUnknown) Error() string {
	return fmt.Sprintf("unknown manifest name=%s tag=%s", err.Name, err.Tag)
}

// ErrManifestUnknownRevision is returned when a manifest cannot be found by
// revision within a repository.
type ErrManifestUnknownRevision struct {
	Name     string
	Revision digest.Digest
}

func (err ErrManifestUnknownRevision) Error() string {
	return fmt.Sprintf("unknown manifest name=%s revision=%s", err.Name, err.Revision)
}

// ErrManifestUnverified is returned when the registry is unable to verify
// the manifest.
type ErrManifestUnverified struct{}

func (ErrManifestUnverified) Error() string {
	return fmt.Sprintf("unverified manifest")
}

// ErrManifestVerification provides a type to collect errors encountered
// during manifest verification. Currently, it accepts errors of all types,
// but it may be narrowed to those involving manifest verification.
type ErrManifestVerification []error

func (errs ErrManifestVerification) Error() string {
	var parts []string
	for _, err := range errs {
		parts = append(parts, err.Error())
	}

	return fmt.Sprintf("errors verifying manifest: %v", strings.Join(parts, ","))
}

// ErrManifestBlobUnknown returned when a referenced blob cannot be found.
type ErrManifestBlobUnknown struct {
	Digest digest.Digest
}

func (err ErrManifestBlobUnknown) Error() string {
	return fmt.Sprintf("unknown blob %v on manifest", err.Digest)
}

// ErrManifestNameInvalid should be used to denote an invalid manifest
// name. Reason may set, indicating the cause of invalidity.
type ErrManifestNameInvalid struct {
	Name   string
	Reason error
}

func (err ErrManifestNameInvalid) Error() string {
	return fmt.Sprintf("manifest name %q invalid: %v", err.Name, err.Reason)
}
