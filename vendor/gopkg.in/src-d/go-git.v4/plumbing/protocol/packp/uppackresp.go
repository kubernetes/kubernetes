package packp

import (
	"errors"
	"io"

	"bufio"

	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

// ErrUploadPackResponseNotDecoded is returned if Read is called without
// decoding first
var ErrUploadPackResponseNotDecoded = errors.New("upload-pack-response should be decoded")

// UploadPackResponse contains all the information responded by the upload-pack
// service, the response implements io.ReadCloser that allows to read the
// packfile directly from it.
type UploadPackResponse struct {
	ShallowUpdate
	ServerResponse

	r          io.ReadCloser
	isShallow  bool
	isMultiACK bool
	isOk       bool
}

// NewUploadPackResponse create a new UploadPackResponse instance, the request
// being responded by the response is required.
func NewUploadPackResponse(req *UploadPackRequest) *UploadPackResponse {
	isShallow := !req.Depth.IsZero()
	isMultiACK := req.Capabilities.Supports(capability.MultiACK) ||
		req.Capabilities.Supports(capability.MultiACKDetailed)

	return &UploadPackResponse{
		isShallow:  isShallow,
		isMultiACK: isMultiACK,
	}
}

// NewUploadPackResponseWithPackfile creates a new UploadPackResponse instance,
// and sets its packfile reader.
func NewUploadPackResponseWithPackfile(req *UploadPackRequest,
	pf io.ReadCloser) *UploadPackResponse {

	r := NewUploadPackResponse(req)
	r.r = pf
	return r
}

// Decode decodes all the responses sent by upload-pack service into the struct
// and prepares it to read the packfile using the Read method
func (r *UploadPackResponse) Decode(reader io.ReadCloser) error {
	buf := bufio.NewReader(reader)

	if r.isShallow {
		if err := r.ShallowUpdate.Decode(buf); err != nil {
			return err
		}
	}

	if err := r.ServerResponse.Decode(buf, r.isMultiACK); err != nil {
		return err
	}

	// now the reader is ready to read the packfile content
	r.r = ioutil.NewReadCloser(buf, reader)

	return nil
}

// Encode encodes an UploadPackResponse.
func (r *UploadPackResponse) Encode(w io.Writer) (err error) {
	if r.isShallow {
		if err := r.ShallowUpdate.Encode(w); err != nil {
			return err
		}
	}

	if err := r.ServerResponse.Encode(w); err != nil {
		return err
	}

	defer ioutil.CheckClose(r.r, &err)
	_, err = io.Copy(w, r.r)
	return err
}

// Read reads the packfile data, if the request was done with any Sideband
// capability the content read should be demultiplexed. If the methods wasn't
// called before the ErrUploadPackResponseNotDecoded will be return
func (r *UploadPackResponse) Read(p []byte) (int, error) {
	if r.r == nil {
		return 0, ErrUploadPackResponseNotDecoded
	}

	return r.r.Read(p)
}

// Close the underlying reader, if any
func (r *UploadPackResponse) Close() error {
	if r.r == nil {
		return nil
	}

	return r.r.Close()
}
