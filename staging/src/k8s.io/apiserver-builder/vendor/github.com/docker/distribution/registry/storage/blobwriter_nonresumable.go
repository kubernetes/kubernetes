// +build noresumabledigest

package storage

import (
	"github.com/docker/distribution/context"
)

// resumeHashAt is a noop when resumable digest support is disabled.
func (bw *blobWriter) resumeDigestAt(ctx context.Context, offset int64) error {
	return errResumableDigestNotAvailable
}

// storeHashState is a noop when resumable digest support is disabled.
func (bw *blobWriter) storeHashState(ctx context.Context) error {
	return errResumableDigestNotAvailable
}
