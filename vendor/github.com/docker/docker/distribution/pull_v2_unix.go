// +build !windows

package distribution

import (
	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
)

func (ld *v2LayerDescriptor) open(ctx context.Context) (distribution.ReadSeekCloser, error) {
	blobs := ld.repo.Blobs(ctx)
	return blobs.Open(ctx, ld.digest)
}
