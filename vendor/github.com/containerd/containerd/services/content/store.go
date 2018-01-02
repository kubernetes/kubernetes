package content

import (
	"context"
	"io"

	contentapi "github.com/containerd/containerd/api/services/content/v1"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	protobuftypes "github.com/gogo/protobuf/types"
	digest "github.com/opencontainers/go-digest"
)

type remoteStore struct {
	client contentapi.ContentClient
}

// NewStoreFromClient returns a new content store
func NewStoreFromClient(client contentapi.ContentClient) content.Store {
	return &remoteStore{
		client: client,
	}
}

func (rs *remoteStore) Info(ctx context.Context, dgst digest.Digest) (content.Info, error) {
	resp, err := rs.client.Info(ctx, &contentapi.InfoRequest{
		Digest: dgst,
	})
	if err != nil {
		return content.Info{}, errdefs.FromGRPC(err)
	}

	return infoFromGRPC(resp.Info), nil
}

func (rs *remoteStore) Walk(ctx context.Context, fn content.WalkFunc, filters ...string) error {
	session, err := rs.client.List(ctx, &contentapi.ListContentRequest{
		Filters: filters,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}

	for {
		msg, err := session.Recv()
		if err != nil {
			if err != io.EOF {
				return errdefs.FromGRPC(err)
			}

			break
		}

		for _, info := range msg.Info {
			if err := fn(infoFromGRPC(info)); err != nil {
				return err
			}
		}
	}

	return nil
}

func (rs *remoteStore) Delete(ctx context.Context, dgst digest.Digest) error {
	if _, err := rs.client.Delete(ctx, &contentapi.DeleteContentRequest{
		Digest: dgst,
	}); err != nil {
		return errdefs.FromGRPC(err)
	}

	return nil
}

func (rs *remoteStore) ReaderAt(ctx context.Context, dgst digest.Digest) (content.ReaderAt, error) {
	i, err := rs.Info(ctx, dgst)
	if err != nil {
		return nil, err
	}

	return &remoteReaderAt{
		ctx:    ctx,
		digest: dgst,
		size:   i.Size,
		client: rs.client,
	}, nil
}

func (rs *remoteStore) Status(ctx context.Context, ref string) (content.Status, error) {
	resp, err := rs.client.Status(ctx, &contentapi.StatusRequest{
		Ref: ref,
	})
	if err != nil {
		return content.Status{}, errdefs.FromGRPC(err)
	}

	status := resp.Status
	return content.Status{
		Ref:       status.Ref,
		StartedAt: status.StartedAt,
		UpdatedAt: status.UpdatedAt,
		Offset:    status.Offset,
		Total:     status.Total,
		Expected:  status.Expected,
	}, nil
}

func (rs *remoteStore) Update(ctx context.Context, info content.Info, fieldpaths ...string) (content.Info, error) {
	resp, err := rs.client.Update(ctx, &contentapi.UpdateRequest{
		Info: infoToGRPC(info),
		UpdateMask: &protobuftypes.FieldMask{
			Paths: fieldpaths,
		},
	})
	if err != nil {
		return content.Info{}, errdefs.FromGRPC(err)
	}
	return infoFromGRPC(resp.Info), nil
}

func (rs *remoteStore) ListStatuses(ctx context.Context, filters ...string) ([]content.Status, error) {
	resp, err := rs.client.ListStatuses(ctx, &contentapi.ListStatusesRequest{
		Filters: filters,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}

	var statuses []content.Status
	for _, status := range resp.Statuses {
		statuses = append(statuses, content.Status{
			Ref:       status.Ref,
			StartedAt: status.StartedAt,
			UpdatedAt: status.UpdatedAt,
			Offset:    status.Offset,
			Total:     status.Total,
			Expected:  status.Expected,
		})
	}

	return statuses, nil
}

func (rs *remoteStore) Writer(ctx context.Context, ref string, size int64, expected digest.Digest) (content.Writer, error) {
	wrclient, offset, err := rs.negotiate(ctx, ref, size, expected)
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}

	return &remoteWriter{
		ref:    ref,
		client: wrclient,
		offset: offset,
	}, nil
}

// Abort implements asynchronous abort. It starts a new write session on the ref l
func (rs *remoteStore) Abort(ctx context.Context, ref string) error {
	if _, err := rs.client.Abort(ctx, &contentapi.AbortRequest{
		Ref: ref,
	}); err != nil {
		return errdefs.FromGRPC(err)
	}

	return nil
}

func (rs *remoteStore) negotiate(ctx context.Context, ref string, size int64, expected digest.Digest) (contentapi.Content_WriteClient, int64, error) {
	wrclient, err := rs.client.Write(ctx)
	if err != nil {
		return nil, 0, err
	}

	if err := wrclient.Send(&contentapi.WriteContentRequest{
		Action:   contentapi.WriteActionStat,
		Ref:      ref,
		Total:    size,
		Expected: expected,
	}); err != nil {
		return nil, 0, err
	}

	resp, err := wrclient.Recv()
	if err != nil {
		return nil, 0, err
	}

	return wrclient, resp.Offset, nil
}

func infoToGRPC(info content.Info) contentapi.Info {
	return contentapi.Info{
		Digest:    info.Digest,
		Size_:     info.Size,
		CreatedAt: info.CreatedAt,
		UpdatedAt: info.UpdatedAt,
		Labels:    info.Labels,
	}
}

func infoFromGRPC(info contentapi.Info) content.Info {
	return content.Info{
		Digest:    info.Digest,
		Size:      info.Size_,
		CreatedAt: info.CreatedAt,
		UpdatedAt: info.UpdatedAt,
		Labels:    info.Labels,
	}
}
