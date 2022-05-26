package http

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/format/pktline"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/go-git/go-git/v5/plumbing/transport"
	"github.com/go-git/go-git/v5/plumbing/transport/internal/common"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

type upSession struct {
	*session
}

func newUploadPackSession(c *http.Client, ep *transport.Endpoint, auth transport.AuthMethod) (transport.UploadPackSession, error) {
	s, err := newSession(c, ep, auth)
	return &upSession{s}, err
}

func (s *upSession) AdvertisedReferences() (*packp.AdvRefs, error) {
	return advertisedReferences(s.session, transport.UploadPackServiceName)
}

func (s *upSession) UploadPack(
	ctx context.Context, req *packp.UploadPackRequest,
) (*packp.UploadPackResponse, error) {

	if req.IsEmpty() {
		return nil, transport.ErrEmptyUploadPackRequest
	}

	if err := req.Validate(); err != nil {
		return nil, err
	}

	url := fmt.Sprintf(
		"%s/%s",
		s.endpoint.String(), transport.UploadPackServiceName,
	)

	content, err := uploadPackRequestToReader(req)
	if err != nil {
		return nil, err
	}

	res, err := s.doRequest(ctx, http.MethodPost, url, content)
	if err != nil {
		return nil, err
	}

	r, err := ioutil.NonEmptyReader(res.Body)
	if err != nil {
		if err == ioutil.ErrEmptyReader || err == io.ErrUnexpectedEOF {
			return nil, transport.ErrEmptyUploadPackRequest
		}

		return nil, err
	}

	rc := ioutil.NewReadCloser(r, res.Body)
	return common.DecodeUploadPackResponse(rc, req)
}

// Close does nothing.
func (s *upSession) Close() error {
	return nil
}

func (s *upSession) doRequest(
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

	applyHeadersToRequest(req, content, s.endpoint.Host, transport.UploadPackServiceName)
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

func uploadPackRequestToReader(req *packp.UploadPackRequest) (*bytes.Buffer, error) {
	buf := bytes.NewBuffer(nil)
	e := pktline.NewEncoder(buf)

	if err := req.UploadRequest.Encode(buf); err != nil {
		return nil, fmt.Errorf("sending upload-req message: %s", err)
	}

	if err := req.UploadHaves.Encode(buf, false); err != nil {
		return nil, fmt.Errorf("sending haves message: %s", err)
	}

	if err := e.EncodeString("done\n"); err != nil {
		return nil, err
	}

	return buf, nil
}
