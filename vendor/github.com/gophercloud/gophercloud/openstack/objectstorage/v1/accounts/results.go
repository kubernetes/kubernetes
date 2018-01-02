package accounts

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/gophercloud/gophercloud"
)

// UpdateResult is returned from a call to the Update function.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// UpdateHeader represents the headers returned in the response from an Update
// request.
type UpdateHeader struct {
	ContentLength int64     `json:"-"`
	ContentType   string    `json:"Content-Type"`
	TransID       string    `json:"X-Trans-Id"`
	Date          time.Time `json:"-"`
}

func (r *UpdateHeader) UnmarshalJSON(b []byte) error {
	type tmp UpdateHeader
	var s struct {
		tmp
		ContentLength string                  `json:"Content-Length"`
		Date          gophercloud.JSONRFC1123 `json:"Date"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = UpdateHeader(s.tmp)

	switch s.ContentLength {
	case "":
		r.ContentLength = 0
	default:
		r.ContentLength, err = strconv.ParseInt(s.ContentLength, 10, 64)
		if err != nil {
			return err
		}
	}

	r.Date = time.Time(s.Date)

	return err
}

// Extract will return a struct of headers returned from a call to Get. To
// obtain a map of headers, call the Extract method on the GetResult.
func (r UpdateResult) Extract() (*UpdateHeader, error) {
	var s *UpdateHeader
	err := r.ExtractInto(&s)
	return s, err
}

// GetHeader represents the headers returned in the response from a Get request.
type GetHeader struct {
	BytesUsed      int64     `json:"-"`
	ContainerCount int64     `json:"-"`
	ContentLength  int64     `json:"-"`
	ObjectCount    int64     `json:"-"`
	ContentType    string    `json:"Content-Type"`
	TransID        string    `json:"X-Trans-Id"`
	TempURLKey     string    `json:"X-Account-Meta-Temp-URL-Key"`
	TempURLKey2    string    `json:"X-Account-Meta-Temp-URL-Key-2"`
	Date           time.Time `json:"-"`
}

func (r *GetHeader) UnmarshalJSON(b []byte) error {
	type tmp GetHeader
	var s struct {
		tmp
		BytesUsed      string `json:"X-Account-Bytes-Used"`
		ContentLength  string `json:"Content-Length"`
		ContainerCount string `json:"X-Account-Container-Count"`
		ObjectCount    string `json:"X-Account-Object-Count"`
		Date           string `json:"Date"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = GetHeader(s.tmp)

	switch s.BytesUsed {
	case "":
		r.BytesUsed = 0
	default:
		r.BytesUsed, err = strconv.ParseInt(s.BytesUsed, 10, 64)
		if err != nil {
			return err
		}
	}

	switch s.ContentLength {
	case "":
		r.ContentLength = 0
	default:
		r.ContentLength, err = strconv.ParseInt(s.ContentLength, 10, 64)
		if err != nil {
			return err
		}
	}

	switch s.ObjectCount {
	case "":
		r.ObjectCount = 0
	default:
		r.ObjectCount, err = strconv.ParseInt(s.ObjectCount, 10, 64)
		if err != nil {
			return err
		}
	}

	switch s.ContainerCount {
	case "":
		r.ContainerCount = 0
	default:
		r.ContainerCount, err = strconv.ParseInt(s.ContainerCount, 10, 64)
		if err != nil {
			return err
		}
	}

	if s.Date != "" {
		r.Date, err = time.Parse(time.RFC1123, s.Date)
	}

	return err
}

// GetResult is returned from a call to the Get function.
type GetResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Get.
func (r GetResult) Extract() (*GetHeader, error) {
	var s *GetHeader
	err := r.ExtractInto(&s)
	return s, err
}

// ExtractMetadata is a function that takes a GetResult (of type *http.Response)
// and returns the custom metatdata associated with the account.
func (r GetResult) ExtractMetadata() (map[string]string, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	metadata := make(map[string]string)
	for k, v := range r.Header {
		if strings.HasPrefix(k, "X-Account-Meta-") {
			key := strings.TrimPrefix(k, "X-Account-Meta-")
			metadata[key] = v[0]
		}
	}
	return metadata, nil
}
