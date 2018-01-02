package content

import (
	"context"
	"io"

	contentapi "github.com/containerd/containerd/api/services/content/v1"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

type remoteWriter struct {
	ref    string
	client contentapi.Content_WriteClient
	offset int64
	digest digest.Digest
}

// send performs a synchronous req-resp cycle on the client.
func (rw *remoteWriter) send(req *contentapi.WriteContentRequest) (*contentapi.WriteContentResponse, error) {
	if err := rw.client.Send(req); err != nil {
		return nil, err
	}

	resp, err := rw.client.Recv()

	if err == nil {
		// try to keep these in sync
		if resp.Digest != "" {
			rw.digest = resp.Digest
		}
	}

	return resp, err
}

func (rw *remoteWriter) Status() (content.Status, error) {
	resp, err := rw.send(&contentapi.WriteContentRequest{
		Action: contentapi.WriteActionStat,
	})
	if err != nil {
		return content.Status{}, err
	}

	return content.Status{
		Ref:       rw.ref,
		Offset:    resp.Offset,
		Total:     resp.Total,
		StartedAt: resp.StartedAt,
		UpdatedAt: resp.UpdatedAt,
	}, nil
}

func (rw *remoteWriter) Digest() digest.Digest {
	return rw.digest
}

func (rw *remoteWriter) Write(p []byte) (n int, err error) {
	offset := rw.offset

	resp, err := rw.send(&contentapi.WriteContentRequest{
		Action: contentapi.WriteActionWrite,
		Offset: offset,
		Data:   p,
	})
	if err != nil {
		return 0, err
	}

	n = int(resp.Offset - offset)
	if n < len(p) {
		err = io.ErrShortWrite
	}

	rw.offset += int64(n)
	if resp.Digest != "" {
		rw.digest = resp.Digest
	}
	return
}

func (rw *remoteWriter) Commit(ctx context.Context, size int64, expected digest.Digest, opts ...content.Opt) error {
	var base content.Info
	for _, opt := range opts {
		if err := opt(&base); err != nil {
			return err
		}
	}
	resp, err := rw.send(&contentapi.WriteContentRequest{
		Action:   contentapi.WriteActionCommit,
		Total:    size,
		Offset:   rw.offset,
		Expected: expected,
		Labels:   base.Labels,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}

	if size != 0 && resp.Offset != size {
		return errors.Errorf("unexpected size: %v != %v", resp.Offset, size)
	}

	if expected != "" && resp.Digest != expected {
		return errors.Errorf("unexpected digest: %v != %v", resp.Digest, expected)
	}

	rw.digest = resp.Digest
	rw.offset = resp.Offset
	return nil
}

func (rw *remoteWriter) Truncate(size int64) error {
	// This truncation won't actually be validated until a write is issued.
	rw.offset = size
	return nil
}

func (rw *remoteWriter) Close() error {
	return rw.client.CloseSend()
}
