package content

import (
	"io"
	"sync"

	api "github.com/containerd/containerd/api/services/content/v1"
	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/metadata"
	"github.com/containerd/containerd/plugin"
	"github.com/golang/protobuf/ptypes/empty"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

type service struct {
	store     content.Store
	publisher events.Publisher
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return make([]byte, 1<<20)
	},
}

var _ api.ContentServer = &service{}

func init() {
	plugin.Register(&plugin.Registration{
		Type: plugin.GRPCPlugin,
		ID:   "content",
		Requires: []plugin.Type{
			plugin.MetadataPlugin,
		},
		InitFn: func(ic *plugin.InitContext) (interface{}, error) {
			m, err := ic.Get(plugin.MetadataPlugin)
			if err != nil {
				return nil, err
			}

			s, err := NewService(m.(*metadata.DB).ContentStore(), ic.Events)
			return s, err
		},
	})
}

// NewService returns the content GRPC server
func NewService(cs content.Store, publisher events.Publisher) (api.ContentServer, error) {
	return &service{
		store:     cs,
		publisher: publisher,
	}, nil
}

func (s *service) Register(server *grpc.Server) error {
	api.RegisterContentServer(server, s)
	return nil
}

func (s *service) Info(ctx context.Context, req *api.InfoRequest) (*api.InfoResponse, error) {
	if err := req.Digest.Validate(); err != nil {
		return nil, grpc.Errorf(codes.InvalidArgument, "%q failed validation", req.Digest)
	}

	bi, err := s.store.Info(ctx, req.Digest)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}

	return &api.InfoResponse{
		Info: infoToGRPC(bi),
	}, nil
}

func (s *service) Update(ctx context.Context, req *api.UpdateRequest) (*api.UpdateResponse, error) {
	if err := req.Info.Digest.Validate(); err != nil {
		return nil, grpc.Errorf(codes.InvalidArgument, "%q failed validation", req.Info.Digest)
	}

	info, err := s.store.Update(ctx, infoFromGRPC(req.Info), req.UpdateMask.GetPaths()...)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}

	return &api.UpdateResponse{
		Info: infoToGRPC(info),
	}, nil
}

func (s *service) List(req *api.ListContentRequest, session api.Content_ListServer) error {
	var (
		buffer    []api.Info
		sendBlock = func(block []api.Info) error {
			// send last block
			return session.Send(&api.ListContentResponse{
				Info: block,
			})
		}
	)

	if err := s.store.Walk(session.Context(), func(info content.Info) error {
		buffer = append(buffer, api.Info{
			Digest:    info.Digest,
			Size_:     info.Size,
			CreatedAt: info.CreatedAt,
			Labels:    info.Labels,
		})

		if len(buffer) >= 100 {
			if err := sendBlock(buffer); err != nil {
				return err
			}

			buffer = buffer[:0]
		}

		return nil
	}, req.Filters...); err != nil {
		return err
	}

	if len(buffer) > 0 {
		// send last block
		if err := sendBlock(buffer); err != nil {
			return err
		}
	}

	return nil
}

func (s *service) Delete(ctx context.Context, req *api.DeleteContentRequest) (*empty.Empty, error) {
	if err := req.Digest.Validate(); err != nil {
		return nil, grpc.Errorf(codes.InvalidArgument, err.Error())
	}

	if err := s.store.Delete(ctx, req.Digest); err != nil {
		return nil, errdefs.ToGRPC(err)
	}

	if err := s.publisher.Publish(ctx, "/content/delete", &eventsapi.ContentDelete{
		Digest: req.Digest,
	}); err != nil {
		return nil, err
	}

	return &empty.Empty{}, nil
}

func (s *service) Read(req *api.ReadContentRequest, session api.Content_ReadServer) error {
	if err := req.Digest.Validate(); err != nil {
		return grpc.Errorf(codes.InvalidArgument, "%v: %v", req.Digest, err)
	}

	oi, err := s.store.Info(session.Context(), req.Digest)
	if err != nil {
		return errdefs.ToGRPC(err)
	}

	ra, err := s.store.ReaderAt(session.Context(), req.Digest)
	if err != nil {
		return errdefs.ToGRPC(err)
	}
	defer ra.Close()

	var (
		offset = req.Offset
		size   = req.Size_

		// TODO(stevvooe): Using the global buffer pool. At 32KB, it is probably
		// little inefficient for work over a fast network. We can tune this later.
		p = bufPool.Get().([]byte)
	)
	defer bufPool.Put(p)

	if offset < 0 {
		offset = 0
	}

	if size <= 0 {
		size = oi.Size - offset
	}

	if offset+size > oi.Size {
		return grpc.Errorf(codes.OutOfRange, "read past object length %v bytes", oi.Size)
	}

	if _, err := io.CopyBuffer(
		&readResponseWriter{session: session},
		io.NewSectionReader(ra, offset, size), p); err != nil {
		return err
	}

	return nil
}

// readResponseWriter is a writer that places the output into ReadContentRequest messages.
//
// This allows io.CopyBuffer to do the heavy lifting of chunking the responses
// into the buffer size.
type readResponseWriter struct {
	offset  int64
	session api.Content_ReadServer
}

func (rw *readResponseWriter) Write(p []byte) (n int, err error) {
	if err := rw.session.Send(&api.ReadContentResponse{
		Offset: rw.offset,
		Data:   p,
	}); err != nil {
		return 0, err
	}

	rw.offset += int64(len(p))
	return len(p), nil
}

func (s *service) Status(ctx context.Context, req *api.StatusRequest) (*api.StatusResponse, error) {
	status, err := s.store.Status(ctx, req.Ref)
	if err != nil {
		return nil, errdefs.ToGRPCf(err, "could not get status for ref %q", req.Ref)
	}

	var resp api.StatusResponse
	resp.Status = &api.Status{
		StartedAt: status.StartedAt,
		UpdatedAt: status.UpdatedAt,
		Ref:       status.Ref,
		Offset:    status.Offset,
		Total:     status.Total,
		Expected:  status.Expected,
	}

	return &resp, nil
}

func (s *service) ListStatuses(ctx context.Context, req *api.ListStatusesRequest) (*api.ListStatusesResponse, error) {
	statuses, err := s.store.ListStatuses(ctx, req.Filters...)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}

	var resp api.ListStatusesResponse
	for _, status := range statuses {
		resp.Statuses = append(resp.Statuses, api.Status{
			StartedAt: status.StartedAt,
			UpdatedAt: status.UpdatedAt,
			Ref:       status.Ref,
			Offset:    status.Offset,
			Total:     status.Total,
			Expected:  status.Expected,
		})
	}

	return &resp, nil
}

func (s *service) Write(session api.Content_WriteServer) (err error) {
	var (
		ctx      = session.Context()
		msg      api.WriteContentResponse
		req      *api.WriteContentRequest
		ref      string
		total    int64
		expected digest.Digest
	)

	defer func(msg *api.WriteContentResponse) {
		// pump through the last message if no error was encountered
		if err != nil {
			if grpc.Code(err) != codes.AlreadyExists {
				// TODO(stevvooe): Really need a log line here to track which
				// errors are actually causing failure on the server side. May want
				// to configure the service with an interceptor to make this work
				// identically across all GRPC methods.
				//
				// This is pretty noisy, so we can remove it but leave it for now.
				log.G(ctx).WithError(err).Error("(*service).Write failed")
			}

			return
		}

		err = session.Send(msg)
	}(&msg)

	// handle the very first request!
	req, err = session.Recv()
	if err != nil {
		return err
	}

	ref = req.Ref

	if ref == "" {
		return grpc.Errorf(codes.InvalidArgument, "first message must have a reference")
	}

	fields := logrus.Fields{
		"ref": ref,
	}
	total = req.Total
	expected = req.Expected
	if total > 0 {
		fields["total"] = total
	}

	if expected != "" {
		fields["expected"] = expected
	}

	ctx = log.WithLogger(ctx, log.G(ctx).WithFields(fields))

	log.G(ctx).Debug("(*service).Write started")
	// this action locks the writer for the session.
	wr, err := s.store.Writer(ctx, ref, total, expected)
	if err != nil {
		return errdefs.ToGRPC(err)
	}
	defer wr.Close()

	for {
		msg.Action = req.Action
		ws, err := wr.Status()
		if err != nil {
			return errdefs.ToGRPC(err)
		}

		msg.Offset = ws.Offset // always set the offset.

		// NOTE(stevvooe): In general, there are two cases underwhich a remote
		// writer is used.
		//
		// For pull, we almost always have this before fetching large content,
		// through descriptors. We allow predeclaration of the expected size
		// and digest.
		//
		// For push, it is more complex. If we want to cut through content into
		// storage, we may have no expectation until we are done processing the
		// content. The case here is the following:
		//
		// 	1. Start writing content.
		// 	2. Compress inline.
		// 	3. Validate digest and size (maybe).
		//
		// Supporting these two paths is quite awkward but it lets both API
		// users use the same writer style for each with a minimum of overhead.
		if req.Expected != "" {
			if expected != "" && expected != req.Expected {
				return grpc.Errorf(codes.InvalidArgument, "inconsistent digest provided: %v != %v", req.Expected, expected)
			}
			expected = req.Expected

			if _, err := s.store.Info(session.Context(), req.Expected); err == nil {
				if err := s.store.Abort(session.Context(), ref); err != nil {
					log.G(ctx).WithError(err).Error("failed to abort write")
				}

				return grpc.Errorf(codes.AlreadyExists, "blob with expected digest %v exists", req.Expected)
			}
		}

		if req.Total > 0 {
			// Update the expected total. Typically, this could be seen at
			// negotiation time or on a commit message.
			if total > 0 && req.Total != total {
				return grpc.Errorf(codes.InvalidArgument, "inconsistent total provided: %v != %v", req.Total, total)
			}
			total = req.Total
		}

		switch req.Action {
		case api.WriteActionStat:
			msg.Digest = wr.Digest()
			msg.StartedAt = ws.StartedAt
			msg.UpdatedAt = ws.UpdatedAt
			msg.Total = total
		case api.WriteActionWrite, api.WriteActionCommit:
			if req.Offset > 0 {
				// validate the offset if provided
				if req.Offset != ws.Offset {
					return grpc.Errorf(codes.OutOfRange, "write @%v must occur at current offset %v", req.Offset, ws.Offset)
				}
			}

			if req.Offset == 0 && ws.Offset > 0 {
				if err := wr.Truncate(req.Offset); err != nil {
					return errors.Wrapf(err, "truncate failed")
				}
				msg.Offset = req.Offset
			}

			// issue the write if we actually have data.
			if len(req.Data) > 0 {
				// While this looks like we could use io.WriterAt here, because we
				// maintain the offset as append only, we just issue the write.
				n, err := wr.Write(req.Data)
				if err != nil {
					return err
				}

				if n != len(req.Data) {
					// TODO(stevvooe): Perhaps, we can recover this by including it
					// in the offset on the write return.
					return grpc.Errorf(codes.DataLoss, "wrote %v of %v bytes", n, len(req.Data))
				}

				msg.Offset += int64(n)
			}

			if req.Action == api.WriteActionCommit {
				var opts []content.Opt
				if req.Labels != nil {
					opts = append(opts, content.WithLabels(req.Labels))
				}
				if err := wr.Commit(ctx, total, expected, opts...); err != nil {
					return err
				}
			}

			msg.Digest = wr.Digest()
		}

		if err := session.Send(&msg); err != nil {
			return err
		}

		req, err = session.Recv()
		if err != nil {
			if err == io.EOF {
				return nil
			}

			return err
		}
	}
}

func (s *service) Abort(ctx context.Context, req *api.AbortRequest) (*empty.Empty, error) {
	if err := s.store.Abort(ctx, req.Ref); err != nil {
		return nil, errdefs.ToGRPC(err)
	}

	return &empty.Empty{}, nil
}
