package distribution

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/reference"
	"github.com/opencontainers/go-digest"
)

var (
	// ErrBlobExists returned when blob already exists
	ErrBlobExists = errors.New("blob exists")

	// ErrBlobDigestUnsupported when blob digest is an unsupported version.
	ErrBlobDigestUnsupported = errors.New("unsupported blob digest")

	// ErrBlobUnknown when blob is not found.
	ErrBlobUnknown = errors.New("unknown blob")

	// ErrBlobUploadUnknown returned when upload is not found.
	ErrBlobUploadUnknown = errors.New("blob upload unknown")

	// ErrBlobInvalidLength returned when the blob has an expected length on
	// commit, meaning mismatched with the descriptor or an invalid value.
	ErrBlobInvalidLength = errors.New("blob invalid length")
)

// ErrBlobInvalidDigest returned when digest check fails.
type ErrBlobInvalidDigest struct {
	Digest digest.Digest
	Reason error
}

func (err ErrBlobInvalidDigest) Error() string {
	return fmt.Sprintf("invalid digest for referenced layer: %v, %v",
		err.Digest, err.Reason)
}

// ErrBlobMounted returned when a blob is mounted from another repository
// instead of initiating an upload session.
type ErrBlobMounted struct {
	From       reference.Canonical
	Descriptor Descriptor
}

func (err ErrBlobMounted) Error() string {
	return fmt.Sprintf("blob mounted from: %v to: %v",
		err.From, err.Descriptor)
}

// Descriptor describes targeted content. Used in conjunction with a blob
// store, a descriptor can be used to fetch, store and target any kind of
// blob. The struct also describes the wire protocol format. Fields should
// only be added but never changed.
type Descriptor struct {
	// MediaType describe the type of the content. All text based formats are
	// encoded as utf-8.
	MediaType string `json:"mediaType,omitempty"`

	// Size in bytes of content.
	Size int64 `json:"size,omitempty"`

	// Digest uniquely identifies the content. A byte stream can be verified
	// against against this digest.
	Digest digest.Digest `json:"digest,omitempty"`

	// URLs contains the source URLs of this content.
	URLs []string `json:"urls,omitempty"`

	// NOTE: Before adding a field here, please ensure that all
	// other options have been exhausted. Much of the type relationships
	// depend on the simplicity of this type.
}

// Descriptor returns the descriptor, to make it satisfy the Describable
// interface. Note that implementations of Describable are generally objects
// which can be described, not simply descriptors; this exception is in place
// to make it more convenient to pass actual descriptors to functions that
// expect Describable objects.
func (d Descriptor) Descriptor() Descriptor {
	return d
}

// BlobStatter makes blob descriptors available by digest. The service may
// provide a descriptor of a different digest if the provided digest is not
// canonical.
type BlobStatter interface {
	// Stat provides metadata about a blob identified by the digest. If the
	// blob is unknown to the describer, ErrBlobUnknown will be returned.
	Stat(ctx context.Context, dgst digest.Digest) (Descriptor, error)
}

// BlobDeleter enables deleting blobs from storage.
type BlobDeleter interface {
	Delete(ctx context.Context, dgst digest.Digest) error
}

// BlobEnumerator enables iterating over blobs from storage
type BlobEnumerator interface {
	Enumerate(ctx context.Context, ingester func(dgst digest.Digest) error) error
}

// BlobDescriptorService manages metadata about a blob by digest. Most
// implementations will not expose such an interface explicitly. Such mappings
// should be maintained by interacting with the BlobIngester. Hence, this is
// left off of BlobService and BlobStore.
type BlobDescriptorService interface {
	BlobStatter

	// SetDescriptor assigns the descriptor to the digest. The provided digest and
	// the digest in the descriptor must map to identical content but they may
	// differ on their algorithm. The descriptor must have the canonical
	// digest of the content and the digest algorithm must match the
	// annotators canonical algorithm.
	//
	// Such a facility can be used to map blobs between digest domains, with
	// the restriction that the algorithm of the descriptor must match the
	// canonical algorithm (ie sha256) of the annotator.
	SetDescriptor(ctx context.Context, dgst digest.Digest, desc Descriptor) error

	// Clear enables descriptors to be unlinked
	Clear(ctx context.Context, dgst digest.Digest) error
}

// BlobDescriptorServiceFactory creates middleware for BlobDescriptorService.
type BlobDescriptorServiceFactory interface {
	BlobAccessController(svc BlobDescriptorService) BlobDescriptorService
}

// ReadSeekCloser is the primary reader type for blob data, combining
// io.ReadSeeker with io.Closer.
type ReadSeekCloser interface {
	io.ReadSeeker
	io.Closer
}

// BlobProvider describes operations for getting blob data.
type BlobProvider interface {
	// Get returns the entire blob identified by digest along with the descriptor.
	Get(ctx context.Context, dgst digest.Digest) ([]byte, error)

	// Open provides a ReadSeekCloser to the blob identified by the provided
	// descriptor. If the blob is not known to the service, an error will be
	// returned.
	Open(ctx context.Context, dgst digest.Digest) (ReadSeekCloser, error)
}

// BlobServer can serve blobs via http.
type BlobServer interface {
	// ServeBlob attempts to serve the blob, identified by dgst, via http. The
	// service may decide to redirect the client elsewhere or serve the data
	// directly.
	//
	// This handler only issues successful responses, such as 2xx or 3xx,
	// meaning it serves data or issues a redirect. If the blob is not
	// available, an error will be returned and the caller may still issue a
	// response.
	//
	// The implementation may serve the same blob from a different digest
	// domain. The appropriate headers will be set for the blob, unless they
	// have already been set by the caller.
	ServeBlob(ctx context.Context, w http.ResponseWriter, r *http.Request, dgst digest.Digest) error
}

// BlobIngester ingests blob data.
type BlobIngester interface {
	// Put inserts the content p into the blob service, returning a descriptor
	// or an error.
	Put(ctx context.Context, mediaType string, p []byte) (Descriptor, error)

	// Create allocates a new blob writer to add a blob to this service. The
	// returned handle can be written to and later resumed using an opaque
	// identifier. With this approach, one can Close and Resume a BlobWriter
	// multiple times until the BlobWriter is committed or cancelled.
	Create(ctx context.Context, options ...BlobCreateOption) (BlobWriter, error)

	// Resume attempts to resume a write to a blob, identified by an id.
	Resume(ctx context.Context, id string) (BlobWriter, error)
}

// BlobCreateOption is a general extensible function argument for blob creation
// methods. A BlobIngester may choose to honor any or none of the given
// BlobCreateOptions, which can be specific to the implementation of the
// BlobIngester receiving them.
// TODO (brianbland): unify this with ManifestServiceOption in the future
type BlobCreateOption interface {
	Apply(interface{}) error
}

// CreateOptions is a collection of blob creation modifiers relevant to general
// blob storage intended to be configured by the BlobCreateOption.Apply method.
type CreateOptions struct {
	Mount struct {
		ShouldMount bool
		From        reference.Canonical
		// Stat allows to pass precalculated descriptor to link and return.
		// Blob access check will be skipped if set.
		Stat *Descriptor
	}
}

// BlobWriter provides a handle for inserting data into a blob store.
// Instances should be obtained from BlobWriteService.Writer and
// BlobWriteService.Resume. If supported by the store, a writer can be
// recovered with the id.
type BlobWriter interface {
	io.WriteCloser
	io.ReaderFrom

	// Size returns the number of bytes written to this blob.
	Size() int64

	// ID returns the identifier for this writer. The ID can be used with the
	// Blob service to later resume the write.
	ID() string

	// StartedAt returns the time this blob write was started.
	StartedAt() time.Time

	// Commit completes the blob writer process. The content is verified
	// against the provided provisional descriptor, which may result in an
	// error. Depending on the implementation, written data may be validated
	// against the provisional descriptor fields. If MediaType is not present,
	// the implementation may reject the commit or assign "application/octet-
	// stream" to the blob. The returned descriptor may have a different
	// digest depending on the blob store, referred to as the canonical
	// descriptor.
	Commit(ctx context.Context, provisional Descriptor) (canonical Descriptor, err error)

	// Cancel ends the blob write without storing any data and frees any
	// associated resources. Any data written thus far will be lost. Cancel
	// implementations should allow multiple calls even after a commit that
	// result in a no-op. This allows use of Cancel in a defer statement,
	// increasing the assurance that it is correctly called.
	Cancel(ctx context.Context) error
}

// BlobService combines the operations to access, read and write blobs. This
// can be used to describe remote blob services.
type BlobService interface {
	BlobStatter
	BlobProvider
	BlobIngester
}

// BlobStore represent the entire suite of blob related operations. Such an
// implementation can access, read, write, delete and serve blobs.
type BlobStore interface {
	BlobService
	BlobServer
	BlobDeleter
}
