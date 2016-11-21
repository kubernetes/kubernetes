package accounts

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/gophercloud/gophercloud"
)

// UpdateResult is returned from a call to the Update function.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// UpdateHeader represents the headers returned in the response from an Update request.
type UpdateHeader struct {
	ContentLength int64                   `json:"-"`
	ContentType   string                  `json:"Content-Type"`
	TransID       string                  `json:"X-Trans-Id"`
	Date          gophercloud.JSONRFC1123 `json:"Date"`
}

func (h *UpdateHeader) UnmarshalJSON(b []byte) error {
	type tmp UpdateHeader
	var updateHeader *struct {
		tmp
		ContentLength string `json:"Content-Length"`
	}
	err := json.Unmarshal(b, &updateHeader)
	if err != nil {
		return err
	}

	*h = UpdateHeader(updateHeader.tmp)

	switch updateHeader.ContentLength {
	case "":
		h.ContentLength = 0
	default:
		h.ContentLength, err = strconv.ParseInt(updateHeader.ContentLength, 10, 64)
		if err != nil {
			return err
		}
	}

	return nil
}

// Extract will return a struct of headers returned from a call to Get. To obtain
// a map of headers, call the ExtractHeader method on the GetResult.
func (ur UpdateResult) Extract() (*UpdateHeader, error) {
	var uh *UpdateHeader
	err := ur.ExtractInto(&uh)
	return uh, err
}

// GetHeader represents the headers returned in the response from a Get request.
type GetHeader struct {
	BytesUsed      int64                   `json:"-"`
	ContainerCount int64                   `json:"-"`
	ContentLength  int64                   `json:"-"`
	ObjectCount    int64                   `json:"-"`
	ContentType    string                  `json:"Content-Type"`
	TransID        string                  `json:"X-Trans-Id"`
	TempURLKey     string                  `json:"X-Account-Meta-Temp-URL-Key"`
	TempURLKey2    string                  `json:"X-Account-Meta-Temp-URL-Key-2"`
	Date           gophercloud.JSONRFC1123 `json:"Date"`
}

func (h *GetHeader) UnmarshalJSON(b []byte) error {
	type tmp GetHeader
	var getHeader *struct {
		tmp
		BytesUsed      string `json:"X-Account-Bytes-Used"`
		ContentLength  string `json:"Content-Length"`
		ContainerCount string `json:"X-Account-Container-Count"`
		ObjectCount    string `json:"X-Account-Object-Count"`
	}
	err := json.Unmarshal(b, &getHeader)
	if err != nil {
		return err
	}

	*h = GetHeader(getHeader.tmp)

	switch getHeader.BytesUsed {
	case "":
		h.BytesUsed = 0
	default:
		h.BytesUsed, err = strconv.ParseInt(getHeader.BytesUsed, 10, 64)
		if err != nil {
			return err
		}
	}

	fmt.Println("getHeader: ", getHeader.ContentLength)
	switch getHeader.ContentLength {
	case "":
		h.ContentLength = 0
	default:
		h.ContentLength, err = strconv.ParseInt(getHeader.ContentLength, 10, 64)
		if err != nil {
			return err
		}
	}

	switch getHeader.ObjectCount {
	case "":
		h.ObjectCount = 0
	default:
		h.ObjectCount, err = strconv.ParseInt(getHeader.ObjectCount, 10, 64)
		if err != nil {
			return err
		}
	}

	switch getHeader.ContainerCount {
	case "":
		h.ContainerCount = 0
	default:
		h.ContainerCount, err = strconv.ParseInt(getHeader.ContainerCount, 10, 64)
		if err != nil {
			return err
		}
	}

	return nil
}

// GetResult is returned from a call to the Get function.
type GetResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Get. To obtain
// a map of headers, call the ExtractHeader method on the GetResult.
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
