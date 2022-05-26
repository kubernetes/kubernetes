// Package server implements the git server protocol. For most use cases, the
// transport-specific implementations should be used.
package server

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/format/packfile"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
	"github.com/go-git/go-git/v5/plumbing/revlist"
	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/plumbing/transport"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

var DefaultServer = NewServer(DefaultLoader)

type server struct {
	loader  Loader
	handler *handler
}

// NewServer returns a transport.Transport implementing a git server,
// independent of transport. Each transport must wrap this.
func NewServer(loader Loader) transport.Transport {
	return &server{
		loader,
		&handler{asClient: false},
	}
}

// NewClient returns a transport.Transport implementing a client with an
// embedded server.
func NewClient(loader Loader) transport.Transport {
	return &server{
		loader,
		&handler{asClient: true},
	}
}

func (s *server) NewUploadPackSession(ep *transport.Endpoint, auth transport.AuthMethod) (transport.UploadPackSession, error) {
	sto, err := s.loader.Load(ep)
	if err != nil {
		return nil, err
	}

	return s.handler.NewUploadPackSession(sto)
}

func (s *server) NewReceivePackSession(ep *transport.Endpoint, auth transport.AuthMethod) (transport.ReceivePackSession, error) {
	sto, err := s.loader.Load(ep)
	if err != nil {
		return nil, err
	}

	return s.handler.NewReceivePackSession(sto)
}

type handler struct {
	asClient bool
}

func (h *handler) NewUploadPackSession(s storer.Storer) (transport.UploadPackSession, error) {
	return &upSession{
		session: session{storer: s, asClient: h.asClient},
	}, nil
}

func (h *handler) NewReceivePackSession(s storer.Storer) (transport.ReceivePackSession, error) {
	return &rpSession{
		session:   session{storer: s, asClient: h.asClient},
		cmdStatus: map[plumbing.ReferenceName]error{},
	}, nil
}

type session struct {
	storer   storer.Storer
	caps     *capability.List
	asClient bool
}

func (s *session) Close() error {
	return nil
}

func (s *session) SetAuth(transport.AuthMethod) error {
	//TODO: deprecate
	return nil
}

func (s *session) checkSupportedCapabilities(cl *capability.List) error {
	for _, c := range cl.All() {
		if !s.caps.Supports(c) {
			return fmt.Errorf("unsupported capability: %s", c)
		}
	}

	return nil
}

type upSession struct {
	session
}

func (s *upSession) AdvertisedReferences() (*packp.AdvRefs, error) {
	ar := packp.NewAdvRefs()

	if err := s.setSupportedCapabilities(ar.Capabilities); err != nil {
		return nil, err
	}

	s.caps = ar.Capabilities

	if err := setReferences(s.storer, ar); err != nil {
		return nil, err
	}

	if err := setHEAD(s.storer, ar); err != nil {
		return nil, err
	}

	if s.asClient && len(ar.References) == 0 {
		return nil, transport.ErrEmptyRemoteRepository
	}

	return ar, nil
}

func (s *upSession) UploadPack(ctx context.Context, req *packp.UploadPackRequest) (*packp.UploadPackResponse, error) {
	if req.IsEmpty() {
		return nil, transport.ErrEmptyUploadPackRequest
	}

	if err := req.Validate(); err != nil {
		return nil, err
	}

	if s.caps == nil {
		s.caps = capability.NewList()
		if err := s.setSupportedCapabilities(s.caps); err != nil {
			return nil, err
		}
	}

	if err := s.checkSupportedCapabilities(req.Capabilities); err != nil {
		return nil, err
	}

	s.caps = req.Capabilities

	if len(req.Shallows) > 0 {
		return nil, fmt.Errorf("shallow not supported")
	}

	objs, err := s.objectsToUpload(req)
	if err != nil {
		return nil, err
	}

	pr, pw := io.Pipe()
	e := packfile.NewEncoder(pw, s.storer, false)
	go func() {
		// TODO: plumb through a pack window.
		_, err := e.Encode(objs, 10)
		pw.CloseWithError(err)
	}()

	return packp.NewUploadPackResponseWithPackfile(req,
		ioutil.NewContextReadCloser(ctx, pr),
	), nil
}

func (s *upSession) objectsToUpload(req *packp.UploadPackRequest) ([]plumbing.Hash, error) {
	haves, err := revlist.Objects(s.storer, req.Haves, nil)
	if err != nil {
		return nil, err
	}

	return revlist.Objects(s.storer, req.Wants, haves)
}

func (*upSession) setSupportedCapabilities(c *capability.List) error {
	if err := c.Set(capability.Agent, capability.DefaultAgent); err != nil {
		return err
	}

	if err := c.Set(capability.OFSDelta); err != nil {
		return err
	}

	return nil
}

type rpSession struct {
	session
	cmdStatus map[plumbing.ReferenceName]error
	firstErr  error
	unpackErr error
}

func (s *rpSession) AdvertisedReferences() (*packp.AdvRefs, error) {
	ar := packp.NewAdvRefs()

	if err := s.setSupportedCapabilities(ar.Capabilities); err != nil {
		return nil, err
	}

	s.caps = ar.Capabilities

	if err := setReferences(s.storer, ar); err != nil {
		return nil, err
	}

	if err := setHEAD(s.storer, ar); err != nil {
		return nil, err
	}

	return ar, nil
}

var (
	ErrUpdateReference = errors.New("failed to update ref")
)

func (s *rpSession) ReceivePack(ctx context.Context, req *packp.ReferenceUpdateRequest) (*packp.ReportStatus, error) {
	if s.caps == nil {
		s.caps = capability.NewList()
		if err := s.setSupportedCapabilities(s.caps); err != nil {
			return nil, err
		}
	}

	if err := s.checkSupportedCapabilities(req.Capabilities); err != nil {
		return nil, err
	}

	s.caps = req.Capabilities

	//TODO: Implement 'atomic' update of references.

	if req.Packfile != nil {
		r := ioutil.NewContextReadCloser(ctx, req.Packfile)
		if err := s.writePackfile(r); err != nil {
			s.unpackErr = err
			s.firstErr = err
			return s.reportStatus(), err
		}
	}

	s.updateReferences(req)
	return s.reportStatus(), s.firstErr
}

func (s *rpSession) updateReferences(req *packp.ReferenceUpdateRequest) {
	for _, cmd := range req.Commands {
		exists, err := referenceExists(s.storer, cmd.Name)
		if err != nil {
			s.setStatus(cmd.Name, err)
			continue
		}

		switch cmd.Action() {
		case packp.Create:
			if exists {
				s.setStatus(cmd.Name, ErrUpdateReference)
				continue
			}

			ref := plumbing.NewHashReference(cmd.Name, cmd.New)
			err := s.storer.SetReference(ref)
			s.setStatus(cmd.Name, err)
		case packp.Delete:
			if !exists {
				s.setStatus(cmd.Name, ErrUpdateReference)
				continue
			}

			err := s.storer.RemoveReference(cmd.Name)
			s.setStatus(cmd.Name, err)
		case packp.Update:
			if !exists {
				s.setStatus(cmd.Name, ErrUpdateReference)
				continue
			}

			ref := plumbing.NewHashReference(cmd.Name, cmd.New)
			err := s.storer.SetReference(ref)
			s.setStatus(cmd.Name, err)
		}
	}
}

func (s *rpSession) writePackfile(r io.ReadCloser) error {
	if r == nil {
		return nil
	}

	if err := packfile.UpdateObjectStorage(s.storer, r); err != nil {
		_ = r.Close()
		return err
	}

	return r.Close()
}

func (s *rpSession) setStatus(ref plumbing.ReferenceName, err error) {
	s.cmdStatus[ref] = err
	if s.firstErr == nil && err != nil {
		s.firstErr = err
	}
}

func (s *rpSession) reportStatus() *packp.ReportStatus {
	if !s.caps.Supports(capability.ReportStatus) {
		return nil
	}

	rs := packp.NewReportStatus()
	rs.UnpackStatus = "ok"

	if s.unpackErr != nil {
		rs.UnpackStatus = s.unpackErr.Error()
	}

	if s.cmdStatus == nil {
		return rs
	}

	for ref, err := range s.cmdStatus {
		msg := "ok"
		if err != nil {
			msg = err.Error()
		}
		status := &packp.CommandStatus{
			ReferenceName: ref,
			Status:        msg,
		}
		rs.CommandStatuses = append(rs.CommandStatuses, status)
	}

	return rs
}

func (*rpSession) setSupportedCapabilities(c *capability.List) error {
	if err := c.Set(capability.Agent, capability.DefaultAgent); err != nil {
		return err
	}

	if err := c.Set(capability.OFSDelta); err != nil {
		return err
	}

	if err := c.Set(capability.DeleteRefs); err != nil {
		return err
	}

	return c.Set(capability.ReportStatus)
}

func setHEAD(s storer.Storer, ar *packp.AdvRefs) error {
	ref, err := s.Reference(plumbing.HEAD)
	if err == plumbing.ErrReferenceNotFound {
		return nil
	}

	if err != nil {
		return err
	}

	if ref.Type() == plumbing.SymbolicReference {
		if err := ar.AddReference(ref); err != nil {
			return nil
		}

		ref, err = storer.ResolveReference(s, ref.Target())
		if err == plumbing.ErrReferenceNotFound {
			return nil
		}

		if err != nil {
			return err
		}
	}

	if ref.Type() != plumbing.HashReference {
		return plumbing.ErrInvalidType
	}

	h := ref.Hash()
	ar.Head = &h

	return nil
}

func setReferences(s storer.Storer, ar *packp.AdvRefs) error {
	//TODO: add peeled references.
	iter, err := s.IterReferences()
	if err != nil {
		return err
	}

	return iter.ForEach(func(ref *plumbing.Reference) error {
		if ref.Type() != plumbing.HashReference {
			return nil
		}

		ar.References[ref.Name().String()] = ref.Hash()
		return nil
	})
}

func referenceExists(s storer.ReferenceStorer, n plumbing.ReferenceName) (bool, error) {
	_, err := s.Reference(n)
	if err == plumbing.ErrReferenceNotFound {
		return false, nil
	}

	return err == nil, err
}
