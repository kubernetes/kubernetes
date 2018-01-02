// +build windows

package distribution

import (
	"net/http"
	"os"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/sirupsen/logrus"
)

var _ distribution.Describable = &v2LayerDescriptor{}

func (ld *v2LayerDescriptor) Descriptor() distribution.Descriptor {
	if ld.src.MediaType == schema2.MediaTypeForeignLayer && len(ld.src.URLs) > 0 {
		return ld.src
	}
	return distribution.Descriptor{}
}

func (ld *v2LayerDescriptor) open(ctx context.Context) (distribution.ReadSeekCloser, error) {
	blobs := ld.repo.Blobs(ctx)
	rsc, err := blobs.Open(ctx, ld.digest)

	if len(ld.src.URLs) == 0 {
		return rsc, err
	}

	// We're done if the registry has this blob.
	if err == nil {
		// Seek does an HTTP GET.  If it succeeds, the blob really is accessible.
		if _, err = rsc.Seek(0, os.SEEK_SET); err == nil {
			return rsc, nil
		}
		rsc.Close()
	}

	// Find the first URL that results in a 200 result code.
	for _, url := range ld.src.URLs {
		logrus.Debugf("Pulling %v from foreign URL %v", ld.digest, url)
		rsc = transport.NewHTTPReadSeeker(http.DefaultClient, url, nil)

		// Seek does an HTTP GET.  If it succeeds, the blob really is accessible.
		_, err = rsc.Seek(0, os.SEEK_SET)
		if err == nil {
			break
		}
		logrus.Debugf("Download for %v failed: %v", ld.digest, err)
		rsc.Close()
		rsc = nil
	}
	return rsc, err
}
