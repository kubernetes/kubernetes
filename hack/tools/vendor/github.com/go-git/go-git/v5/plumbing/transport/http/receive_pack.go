package http

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/sideband"
	"github.com/go-git/go-git/v5/plumbing/transport"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

type rpSession struct {
	*session
}

func newReceivePackSession(c *http.Client, ep *transport.Endpoint, auth transport.AuthMethod) (transport.ReceivePackSession, error) {
	s, err := newSession(c, ep, auth)
	return &rpSession{s}, err
}

func (s *rpSession) AdvertisedReferences() (*packp.AdvRefs, error) {
	return advertisedReferences(s.session, transport.ReceivePackServiceName)
}

func (s *rpSession) ReceivePack(ctx context.Context, req *packp.ReferenceUpdateRequest) (
	*packp.ReportStatus, error) {
	url := fmt.Sprintf(
		"%s/%s",
		s.endpoint.String(), transport.ReceivePackServiceName,
	)

	buf := bytes.NewBuffer(nil)
	if err := req.Encode(buf); err != nil {
		return nil, err
	}

	res, err := s.doRequest(ctx, http.MethodPost, url, buf)
	if err != nil {
		return nil, err
	}

	r, err := ioutil.NonEmptyReader(res.Body)
	if err == ioutil.ErrEmptyReader {
		return nil, nil
	}

	if err != nil {
		return nil, err
	}

	var d *sideband.Demuxer
	if req.Capabilities.Supports(capability.Sideband64k) {
		d = sideband.NewDemuxer(sideband.Sideband64k, r)
	} else if req.Capabilities.Supports(capability.Sideband) {
		d = sideband.NewDemuxer(sideband.Sideband, r)
	}
	if d != nil {
		d.Progress = req.Progress
		r = d
	}

	rc := ioutil.NewReadCloser(r, res.Body)

	report := packp.NewReportStatus()
	if err := report.Decode(rc); err != nil {
		return nil, err
	}

	return report, report.Error()
}

func (s *rpSession) doRequest(
	ctx context.Context, method, url string, content *bytes.Buffer,
) (*http.Response, error) {

	var body io.Reader
	if content != nil {
		body = content
	}

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, plumbing.NewPermanentError(err)
	}

	applyHeadersToRequest(req, content, s.endpoint.Host, transport.ReceivePackServiceName)
	s.ApplyAuthToRequest(req)

	res, err := s.client.Do(req.WithContext(ctx))
	if err != nil {
		return nil, plumbing.NewUnexpectedError(err)
	}

	if err := NewErr(res); err != nil {
		_ = res.Body.Close()
		return nil, err
	}

	return res, nil
}
