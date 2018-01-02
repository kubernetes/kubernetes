package remotecontext

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"regexp"

	"github.com/docker/docker/builder"
	"github.com/pkg/errors"
)

// When downloading remote contexts, limit the amount (in bytes)
// to be read from the response body in order to detect its Content-Type
const maxPreambleLength = 100

const acceptableRemoteMIME = `(?:application/(?:(?:x\-)?tar|octet\-stream|((?:x\-)?(?:gzip|bzip2?|xz)))|(?:text/plain))`

var mimeRe = regexp.MustCompile(acceptableRemoteMIME)

// MakeRemoteContext downloads a context from remoteURL and returns it.
//
// If contentTypeHandlers is non-nil, then the Content-Type header is read along with a maximum of
// maxPreambleLength bytes from the body to help detecting the MIME type.
// Look at acceptableRemoteMIME for more details.
//
// If a match is found, then the body is sent to the contentType handler and a (potentially compressed) tar stream is expected
// to be returned. If no match is found, it is assumed the body is a tar stream (compressed or not).
// In either case, an (assumed) tar stream is passed to FromArchive whose result is returned.
func MakeRemoteContext(remoteURL string, contentTypeHandlers map[string]func(io.ReadCloser) (io.ReadCloser, error)) (builder.Source, error) {
	f, err := GetWithStatusError(remoteURL)
	if err != nil {
		return nil, fmt.Errorf("error downloading remote context %s: %v", remoteURL, err)
	}
	defer f.Body.Close()

	var contextReader io.ReadCloser
	if contentTypeHandlers != nil {
		contentType := f.Header.Get("Content-Type")
		clen := f.ContentLength

		contentType, contextReader, err = inspectResponse(contentType, f.Body, clen)
		if err != nil {
			return nil, fmt.Errorf("error detecting content type for remote %s: %v", remoteURL, err)
		}
		defer contextReader.Close()

		// This loop tries to find a content-type handler for the detected content-type.
		// If it could not find one from the caller-supplied map, it tries the empty content-type `""`
		// which is interpreted as a fallback handler (usually used for raw tar contexts).
		for _, ct := range []string{contentType, ""} {
			if fn, ok := contentTypeHandlers[ct]; ok {
				defer contextReader.Close()
				if contextReader, err = fn(contextReader); err != nil {
					return nil, err
				}
				break
			}
		}
	}

	// Pass through - this is a pre-packaged context, presumably
	// with a Dockerfile with the right name inside it.
	return FromArchive(contextReader)
}

// GetWithStatusError does an http.Get() and returns an error if the
// status code is 4xx or 5xx.
func GetWithStatusError(url string) (resp *http.Response, err error) {
	if resp, err = http.Get(url); err != nil {
		return nil, err
	}
	if resp.StatusCode < 400 {
		return resp, nil
	}
	msg := fmt.Sprintf("failed to GET %s with status %s", url, resp.Status)
	body, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, errors.Wrapf(err, msg+": error reading body")
	}
	return nil, errors.Errorf(msg+": %s", bytes.TrimSpace(body))
}

// inspectResponse looks into the http response data at r to determine whether its
// content-type is on the list of acceptable content types for remote build contexts.
// This function returns:
//    - a string representation of the detected content-type
//    - an io.Reader for the response body
//    - an error value which will be non-nil either when something goes wrong while
//      reading bytes from r or when the detected content-type is not acceptable.
func inspectResponse(ct string, r io.ReadCloser, clen int64) (string, io.ReadCloser, error) {
	plen := clen
	if plen <= 0 || plen > maxPreambleLength {
		plen = maxPreambleLength
	}

	preamble := make([]byte, plen, plen)
	rlen, err := r.Read(preamble)
	if rlen == 0 {
		return ct, r, errors.New("empty response")
	}
	if err != nil && err != io.EOF {
		return ct, r, err
	}

	preambleR := bytes.NewReader(preamble[:rlen])
	bodyReader := ioutil.NopCloser(io.MultiReader(preambleR, r))
	// Some web servers will use application/octet-stream as the default
	// content type for files without an extension (e.g. 'Dockerfile')
	// so if we receive this value we better check for text content
	contentType := ct
	if len(ct) == 0 || ct == mimeTypes.OctetStream {
		contentType, _, err = detectContentType(preamble)
		if err != nil {
			return contentType, bodyReader, err
		}
	}

	contentType = selectAcceptableMIME(contentType)
	var cterr error
	if len(contentType) == 0 {
		cterr = fmt.Errorf("unsupported Content-Type %q", ct)
		contentType = ct
	}

	return contentType, bodyReader, cterr
}

func selectAcceptableMIME(ct string) string {
	return mimeRe.FindString(ct)
}
