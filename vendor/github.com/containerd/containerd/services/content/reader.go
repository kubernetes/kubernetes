package content

import (
	"context"

	contentapi "github.com/containerd/containerd/api/services/content/v1"
	digest "github.com/opencontainers/go-digest"
)

type remoteReaderAt struct {
	ctx    context.Context
	digest digest.Digest
	size   int64
	client contentapi.ContentClient
}

func (ra *remoteReaderAt) Size() int64 {
	return ra.size
}

func (ra *remoteReaderAt) ReadAt(p []byte, off int64) (n int, err error) {
	rr := &contentapi.ReadContentRequest{
		Digest: ra.digest,
		Offset: off,
		Size_:  int64(len(p)),
	}
	rc, err := ra.client.Read(ra.ctx, rr)
	if err != nil {
		return 0, err
	}

	for len(p) > 0 {
		var resp *contentapi.ReadContentResponse
		// fill our buffer up until we can fill p.
		resp, err = rc.Recv()
		if err != nil {
			return n, err
		}

		copied := copy(p, resp.Data)
		n += copied
		p = p[copied:]
	}
	return n, nil
}

func (ra *remoteReaderAt) Close() error {
	return nil
}
