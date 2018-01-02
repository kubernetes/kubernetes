// Package test implements common test suite for different transport
// implementations.
//
package test

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/storage/memory"

	. "gopkg.in/check.v1"
)

type ReceivePackSuite struct {
	Endpoint            transport.Endpoint
	EmptyEndpoint       transport.Endpoint
	NonExistentEndpoint transport.Endpoint
	EmptyAuth           transport.AuthMethod
	Client              transport.Transport
}

func (s *ReceivePackSuite) TestAdvertisedReferencesEmpty(c *C) {
	r, err := s.Client.NewReceivePackSession(s.EmptyEndpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	defer func() { c.Assert(r.Close(), IsNil) }()

	ar, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)
	c.Assert(ar.Head, IsNil)
}

func (s *ReceivePackSuite) TestAdvertisedReferencesNotExists(c *C) {
	r, err := s.Client.NewReceivePackSession(s.NonExistentEndpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	ar, err := r.AdvertisedReferences()
	c.Assert(err, Equals, transport.ErrRepositoryNotFound)
	c.Assert(ar, IsNil)
	c.Assert(r.Close(), IsNil)

	r, err = s.Client.NewReceivePackSession(s.NonExistentEndpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"master", plumbing.ZeroHash, plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5")},
	}

	writer, err := r.ReceivePack(context.Background(), req)
	c.Assert(err, Equals, transport.ErrRepositoryNotFound)
	c.Assert(writer, IsNil)
	c.Assert(r.Close(), IsNil)
}

func (s *ReceivePackSuite) TestCallAdvertisedReferenceTwice(c *C) {
	r, err := s.Client.NewReceivePackSession(s.Endpoint, s.EmptyAuth)
	defer func() { c.Assert(r.Close(), IsNil) }()
	c.Assert(err, IsNil)
	ar1, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)
	c.Assert(ar1, NotNil)
	ar2, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)
	c.Assert(ar2, DeepEquals, ar1)
}

func (s *ReceivePackSuite) TestDefaultBranch(c *C) {
	r, err := s.Client.NewReceivePackSession(s.Endpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	defer func() { c.Assert(r.Close(), IsNil) }()

	info, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)
	ref, ok := info.References["refs/heads/master"]
	c.Assert(ok, Equals, true)
	c.Assert(ref, Equals, fixtures.Basic().One().Head)
}

func (s *ReceivePackSuite) TestCapabilities(c *C) {
	r, err := s.Client.NewReceivePackSession(s.Endpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	defer func() { c.Assert(r.Close(), IsNil) }()

	info, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)
	c.Assert(info.Capabilities.Get("agent"), HasLen, 1)
}

func (s *ReceivePackSuite) TestFullSendPackOnEmpty(c *C) {
	endpoint := s.EmptyEndpoint
	full := true
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/master", plumbing.ZeroHash, fixture.Head},
	}
	s.receivePack(c, endpoint, req, fixture, full)
	s.checkRemoteHead(c, endpoint, fixture.Head)
}

func (s *ReceivePackSuite) TestSendPackWithContext(c *C) {
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Packfile = fixture.Packfile()
	req.Commands = []*packp.Command{
		{"refs/heads/master", plumbing.ZeroHash, fixture.Head},
	}

	r, err := s.Client.NewReceivePackSession(s.EmptyEndpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	defer func() { c.Assert(r.Close(), IsNil) }()

	info, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)
	c.Assert(info, NotNil)

	ctx, close := context.WithCancel(context.TODO())
	close()

	report, err := r.ReceivePack(ctx, req)
	c.Assert(err, NotNil)
	c.Assert(report, IsNil)
}

func (s *ReceivePackSuite) TestSendPackOnEmpty(c *C) {
	endpoint := s.EmptyEndpoint
	full := false
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/master", plumbing.ZeroHash, fixture.Head},
	}
	s.receivePack(c, endpoint, req, fixture, full)
	s.checkRemoteHead(c, endpoint, fixture.Head)
}

func (s *ReceivePackSuite) TestSendPackOnEmptyWithReportStatus(c *C) {
	endpoint := s.EmptyEndpoint
	full := false
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/master", plumbing.ZeroHash, fixture.Head},
	}
	req.Capabilities.Set(capability.ReportStatus)
	s.receivePack(c, endpoint, req, fixture, full)
	s.checkRemoteHead(c, endpoint, fixture.Head)
}

func (s *ReceivePackSuite) TestFullSendPackOnNonEmpty(c *C) {
	endpoint := s.Endpoint
	full := true
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/master", fixture.Head, fixture.Head},
	}
	s.receivePack(c, endpoint, req, fixture, full)
	s.checkRemoteHead(c, endpoint, fixture.Head)
}

func (s *ReceivePackSuite) TestSendPackOnNonEmpty(c *C) {
	endpoint := s.Endpoint
	full := false
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/master", fixture.Head, fixture.Head},
	}
	s.receivePack(c, endpoint, req, fixture, full)
	s.checkRemoteHead(c, endpoint, fixture.Head)
}

func (s *ReceivePackSuite) TestSendPackOnNonEmptyWithReportStatus(c *C) {
	endpoint := s.Endpoint
	full := false
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/master", fixture.Head, fixture.Head},
	}
	req.Capabilities.Set(capability.ReportStatus)

	s.receivePack(c, endpoint, req, fixture, full)
	s.checkRemoteHead(c, endpoint, fixture.Head)
}

func (s *ReceivePackSuite) TestSendPackOnNonEmptyWithReportStatusWithError(c *C) {
	endpoint := s.Endpoint
	full := false
	fixture := fixtures.Basic().ByTag("packfile").One()
	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/master", plumbing.ZeroHash, fixture.Head},
	}
	req.Capabilities.Set(capability.ReportStatus)

	report, err := s.receivePackNoCheck(c, endpoint, req, fixture, full)
	//XXX: Recent git versions return "failed to update ref", while older
	//     (>=1.9) return "failed to lock".
	c.Assert(err, ErrorMatches, ".*(failed to update ref|failed to lock).*")
	c.Assert(report.UnpackStatus, Equals, "ok")
	c.Assert(len(report.CommandStatuses), Equals, 1)
	c.Assert(report.CommandStatuses[0].ReferenceName, Equals, plumbing.ReferenceName("refs/heads/master"))
	c.Assert(report.CommandStatuses[0].Status, Matches, "(failed to update ref|failed to lock)")
	s.checkRemoteHead(c, endpoint, fixture.Head)
}

func (s *ReceivePackSuite) receivePackNoCheck(c *C, ep transport.Endpoint,
	req *packp.ReferenceUpdateRequest, fixture *fixtures.Fixture,
	callAdvertisedReferences bool) (*packp.ReportStatus, error) {
	url := ""
	if fixture != nil {
		url = fixture.URL
	}
	comment := Commentf(
		"failed with ep=%s fixture=%s callAdvertisedReferences=%s",
		ep.String(), url, callAdvertisedReferences,
	)

	r, err := s.Client.NewReceivePackSession(ep, s.EmptyAuth)
	c.Assert(err, IsNil, comment)
	defer func() { c.Assert(r.Close(), IsNil, comment) }()

	if callAdvertisedReferences {
		info, err := r.AdvertisedReferences()
		c.Assert(err, IsNil, comment)
		c.Assert(info, NotNil, comment)
	}

	if fixture != nil {
		c.Assert(fixture.Packfile(), NotNil)
		req.Packfile = fixture.Packfile()
	} else {
		req.Packfile = s.emptyPackfile()
	}

	return r.ReceivePack(context.Background(), req)
}

func (s *ReceivePackSuite) receivePack(c *C, ep transport.Endpoint,
	req *packp.ReferenceUpdateRequest, fixture *fixtures.Fixture,
	callAdvertisedReferences bool) {

	url := ""
	if fixture != nil {
		url = fixture.URL
	}

	comment := Commentf(
		"failed with ep=%s fixture=%s callAdvertisedReferences=%s",
		ep.String(), url, callAdvertisedReferences,
	)
	report, err := s.receivePackNoCheck(c, ep, req, fixture, callAdvertisedReferences)

	c.Assert(err, IsNil, comment)
	if req.Capabilities.Supports(capability.ReportStatus) {
		c.Assert(report, NotNil, comment)
		c.Assert(report.Error(), IsNil, comment)
	} else {
		c.Assert(report, IsNil, comment)
	}
}

func (s *ReceivePackSuite) checkRemoteHead(c *C, ep transport.Endpoint, head plumbing.Hash) {
	s.checkRemoteReference(c, ep, "refs/heads/master", head)
}

func (s *ReceivePackSuite) checkRemoteReference(c *C, ep transport.Endpoint,
	refName string, head plumbing.Hash) {

	r, err := s.Client.NewUploadPackSession(ep, s.EmptyAuth)
	c.Assert(err, IsNil)
	defer func() { c.Assert(r.Close(), IsNil) }()
	ar, err := r.AdvertisedReferences()
	c.Assert(err, IsNil, Commentf("endpoint: %s", ep.String()))
	ref, ok := ar.References[refName]
	if head == plumbing.ZeroHash {
		c.Assert(ok, Equals, false)
	} else {
		c.Assert(ok, Equals, true)
		c.Assert(ref, DeepEquals, head)
	}
}

func (s *ReceivePackSuite) TestSendPackAddDeleteReference(c *C) {
	s.testSendPackAddReference(c)
	s.testSendPackDeleteReference(c)
}

func (s *ReceivePackSuite) testSendPackAddReference(c *C) {
	r, err := s.Client.NewReceivePackSession(s.Endpoint, s.EmptyAuth)
	c.Assert(err, IsNil)

	fixture := fixtures.Basic().ByTag("packfile").One()

	ar, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)

	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/newbranch", plumbing.ZeroHash, fixture.Head},
	}
	if ar.Capabilities.Supports(capability.ReportStatus) {
		req.Capabilities.Set(capability.ReportStatus)
	}

	c.Assert(r.Close(), IsNil)

	s.receivePack(c, s.Endpoint, req, nil, false)
	s.checkRemoteReference(c, s.Endpoint, "refs/heads/newbranch", fixture.Head)
}

func (s *ReceivePackSuite) testSendPackDeleteReference(c *C) {
	r, err := s.Client.NewReceivePackSession(s.Endpoint, s.EmptyAuth)
	c.Assert(err, IsNil)

	fixture := fixtures.Basic().ByTag("packfile").One()

	ar, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)

	req := packp.NewReferenceUpdateRequest()
	req.Commands = []*packp.Command{
		{"refs/heads/newbranch", fixture.Head, plumbing.ZeroHash},
	}
	if ar.Capabilities.Supports(capability.ReportStatus) {
		req.Capabilities.Set(capability.ReportStatus)
	}

	if !ar.Capabilities.Supports(capability.DeleteRefs) {
		c.Fatal("capability delete-refs not supported")
	}

	c.Assert(r.Close(), IsNil)

	s.receivePack(c, s.Endpoint, req, nil, false)
	s.checkRemoteReference(c, s.Endpoint, "refs/heads/newbranch", plumbing.ZeroHash)
}

func (s *ReceivePackSuite) emptyPackfile() io.ReadCloser {
	var buf bytes.Buffer
	e := packfile.NewEncoder(&buf, memory.NewStorage(), false)
	_, err := e.Encode(nil)
	if err != nil {
		panic(err)
	}

	return ioutil.NopCloser(&buf)
}
