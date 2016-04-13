package httputils

import (
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/docker/docker/pkg/jsonmessage"
)

// Download requests a given URL and returns an io.Reader
func Download(url string) (resp *http.Response, err error) {
	if resp, err = http.Get(url); err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("Got HTTP status code >= 400: %s", resp.Status)
	}
	return resp, nil
}

// NewHTTPRequestError returns a JSON response error
func NewHTTPRequestError(msg string, res *http.Response) error {
	return &jsonmessage.JSONError{
		Message: msg,
		Code:    res.StatusCode,
	}
}

type ServerHeader struct {
	App string // docker
	Ver string // 1.8.0-dev
	OS  string // windows or linux
}

// parseServerHeader extracts pieces from am HTTP server header
// which is in the format "docker/version (os)" eg docker/1.8.0-dev (windows)
func ParseServerHeader(hdr string) (*ServerHeader, error) {
	re := regexp.MustCompile(`.*\((.+)\).*$`)
	r := &ServerHeader{}
	if matches := re.FindStringSubmatch(hdr); matches != nil {
		r.OS = matches[1]
		parts := strings.Split(hdr, "/")
		if len(parts) != 2 {
			return nil, errors.New("Bad header: '/' missing")
		}
		r.App = parts[0]
		v := strings.Split(parts[1], " ")
		if len(v) != 2 {
			return nil, errors.New("Bad header: Expected single space")
		}
		r.Ver = v[0]
		return r, nil
	}
	return nil, errors.New("Bad header: Failed regex match")
}
