package distribution

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/registry"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

// Pusher is an interface that abstracts pushing for different API versions.
type Pusher interface {
	// Push tries to push the image configured at the creation of Pusher.
	// Push returns an error if any, as well as a boolean that determines whether to retry Push on the next configured endpoint.
	//
	// TODO(tiborvass): have Push() take a reference to repository + tag, so that the pusher itself is repository-agnostic.
	Push(ctx context.Context) error
}

const compressionBufSize = 32768

// NewPusher creates a new Pusher interface that will push to either a v1 or v2
// registry. The endpoint argument contains a Version field that determines
// whether a v1 or v2 pusher will be created. The other parameters are passed
// through to the underlying pusher implementation for use during the actual
// push operation.
func NewPusher(ref reference.Named, endpoint registry.APIEndpoint, repoInfo *registry.RepositoryInfo, imagePushConfig *ImagePushConfig) (Pusher, error) {
	switch endpoint.Version {
	case registry.APIVersion2:
		return &v2Pusher{
			v2MetadataService: metadata.NewV2MetadataService(imagePushConfig.MetadataStore),
			ref:               ref,
			endpoint:          endpoint,
			repoInfo:          repoInfo,
			config:            imagePushConfig,
		}, nil
	case registry.APIVersion1:
		return &v1Pusher{
			v1IDService: metadata.NewV1IDService(imagePushConfig.MetadataStore),
			ref:         ref,
			endpoint:    endpoint,
			repoInfo:    repoInfo,
			config:      imagePushConfig,
		}, nil
	}
	return nil, fmt.Errorf("unknown version %d for registry %s", endpoint.Version, endpoint.URL)
}

// Push initiates a push operation on ref.
// ref is the specific variant of the image to be pushed.
// If no tag is provided, all tags will be pushed.
func Push(ctx context.Context, ref reference.Named, imagePushConfig *ImagePushConfig) error {
	// FIXME: Allow to interrupt current push when new push of same image is done.

	// Resolve the Repository name from fqn to RepositoryInfo
	repoInfo, err := imagePushConfig.RegistryService.ResolveRepository(ref)
	if err != nil {
		return err
	}

	endpoints, err := imagePushConfig.RegistryService.LookupPushEndpoints(reference.Domain(repoInfo.Name))
	if err != nil {
		return err
	}

	progress.Messagef(imagePushConfig.ProgressOutput, "", "The push refers to a repository [%s]", repoInfo.Name.Name())

	associations := imagePushConfig.ReferenceStore.ReferencesByName(repoInfo.Name)
	if len(associations) == 0 {
		return fmt.Errorf("An image does not exist locally with the tag: %s", reference.FamiliarName(repoInfo.Name))
	}

	var (
		lastErr error

		// confirmedV2 is set to true if a push attempt managed to
		// confirm that it was talking to a v2 registry. This will
		// prevent fallback to the v1 protocol.
		confirmedV2 bool

		// confirmedTLSRegistries is a map indicating which registries
		// are known to be using TLS. There should never be a plaintext
		// retry for any of these.
		confirmedTLSRegistries = make(map[string]struct{})
	)

	for _, endpoint := range endpoints {
		if imagePushConfig.RequireSchema2 && endpoint.Version == registry.APIVersion1 {
			continue
		}
		if confirmedV2 && endpoint.Version == registry.APIVersion1 {
			logrus.Debugf("Skipping v1 endpoint %s because v2 registry was detected", endpoint.URL)
			continue
		}

		if endpoint.URL.Scheme != "https" {
			if _, confirmedTLS := confirmedTLSRegistries[endpoint.URL.Host]; confirmedTLS {
				logrus.Debugf("Skipping non-TLS endpoint %s for host/port that appears to use TLS", endpoint.URL)
				continue
			}
		}

		logrus.Debugf("Trying to push %s to %s %s", repoInfo.Name.Name(), endpoint.URL, endpoint.Version)

		pusher, err := NewPusher(ref, endpoint, repoInfo, imagePushConfig)
		if err != nil {
			lastErr = err
			continue
		}
		if err := pusher.Push(ctx); err != nil {
			// Was this push cancelled? If so, don't try to fall
			// back.
			select {
			case <-ctx.Done():
			default:
				if fallbackErr, ok := err.(fallbackError); ok {
					confirmedV2 = confirmedV2 || fallbackErr.confirmedV2
					if fallbackErr.transportOK && endpoint.URL.Scheme == "https" {
						confirmedTLSRegistries[endpoint.URL.Host] = struct{}{}
					}
					err = fallbackErr.err
					lastErr = err
					logrus.Infof("Attempting next endpoint for push after error: %v", err)
					continue
				}
			}

			logrus.Errorf("Not continuing with push after error: %v", err)
			return err
		}

		imagePushConfig.ImageEventLogger(reference.FamiliarString(ref), reference.FamiliarName(repoInfo.Name), "push")
		return nil
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("no endpoints found for %s", repoInfo.Name.Name())
	}
	return lastErr
}

// compress returns an io.ReadCloser which will supply a compressed version of
// the provided Reader. The caller must close the ReadCloser after reading the
// compressed data.
//
// Note that this function returns a reader instead of taking a writer as an
// argument so that it can be used with httpBlobWriter's ReadFrom method.
// Using httpBlobWriter's Write method would send a PATCH request for every
// Write call.
//
// The second return value is a channel that gets closed when the goroutine
// is finished. This allows the caller to make sure the goroutine finishes
// before it releases any resources connected with the reader that was
// passed in.
func compress(in io.Reader) (io.ReadCloser, chan struct{}) {
	compressionDone := make(chan struct{})

	pipeReader, pipeWriter := io.Pipe()
	// Use a bufio.Writer to avoid excessive chunking in HTTP request.
	bufWriter := bufio.NewWriterSize(pipeWriter, compressionBufSize)
	compressor := gzip.NewWriter(bufWriter)

	go func() {
		_, err := io.Copy(compressor, in)
		if err == nil {
			err = compressor.Close()
		}
		if err == nil {
			err = bufWriter.Flush()
		}
		if err != nil {
			pipeWriter.CloseWithError(err)
		} else {
			pipeWriter.Close()
		}
		close(compressionDone)
	}()

	return pipeReader, compressionDone
}
