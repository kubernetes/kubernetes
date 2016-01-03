package accounts

import (
	"strings"
	"time"

	"github.com/rackspace/gophercloud"
)

// UpdateResult is returned from a call to the Update function.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// UpdateHeader represents the headers returned in the response from an Update request.
type UpdateHeader struct {
	ContentLength string    `mapstructure:"Content-Length"`
	ContentType   string    `mapstructure:"Content-Type"`
	Date          time.Time `mapstructure:"-"`
	TransID       string    `mapstructure:"X-Trans-Id"`
}

// Extract will return a struct of headers returned from a call to Get. To obtain
// a map of headers, call the ExtractHeader method on the GetResult.
func (ur UpdateResult) Extract() (UpdateHeader, error) {
	var uh UpdateHeader
	if ur.Err != nil {
		return uh, ur.Err
	}

	if err := gophercloud.DecodeHeader(ur.Header, &uh); err != nil {
		return uh, err
	}

	if date, ok := ur.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, ur.Header["Date"][0])
		if err != nil {
			return uh, err
		}
		uh.Date = t
	}

	return uh, nil
}

// GetHeader represents the headers returned in the response from a Get request.
type GetHeader struct {
	BytesUsed      int64     `mapstructure:"X-Account-Bytes-Used"`
	ContainerCount int       `mapstructure:"X-Account-Container-Count"`
	ContentLength  int64     `mapstructure:"Content-Length"`
	ContentType    string    `mapstructure:"Content-Type"`
	Date           time.Time `mapstructure:"-"`
	ObjectCount    int64     `mapstructure:"X-Account-Object-Count"`
	TransID        string    `mapstructure:"X-Trans-Id"`
	TempURLKey     string    `mapstructure:"X-Account-Meta-Temp-URL-Key"`
	TempURLKey2    string    `mapstructure:"X-Account-Meta-Temp-URL-Key-2"`
}

// GetResult is returned from a call to the Get function.
type GetResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Get. To obtain
// a map of headers, call the ExtractHeader method on the GetResult.
func (gr GetResult) Extract() (GetHeader, error) {
	var gh GetHeader
	if gr.Err != nil {
		return gh, gr.Err
	}

	if err := gophercloud.DecodeHeader(gr.Header, &gh); err != nil {
		return gh, err
	}

	if date, ok := gr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, gr.Header["Date"][0])
		if err != nil {
			return gh, err
		}
		gh.Date = t
	}

	return gh, nil
}

// ExtractMetadata is a function that takes a GetResult (of type *http.Response)
// and returns the custom metatdata associated with the account.
func (gr GetResult) ExtractMetadata() (map[string]string, error) {
	if gr.Err != nil {
		return nil, gr.Err
	}

	metadata := make(map[string]string)
	for k, v := range gr.Header {
		if strings.HasPrefix(k, "X-Account-Meta-") {
			key := strings.TrimPrefix(k, "X-Account-Meta-")
			metadata[key] = v[0]
		}
	}
	return metadata, nil
}
