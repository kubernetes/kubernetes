package containers

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Container represents a container resource.
type Container struct {
	// The total number of bytes stored in the container.
	Bytes int64 `json:"bytes"`

	// The total number of objects stored in the container.
	Count int64 `json:"count"`

	// The name of the container.
	Name string `json:"name"`
}

// ContainerPage is the page returned by a pager when traversing over a
// collection of containers.
type ContainerPage struct {
	pagination.MarkerPageBase
}

//IsEmpty returns true if a ListResult contains no container names.
func (r ContainerPage) IsEmpty() (bool, error) {
	names, err := ExtractNames(r)
	return len(names) == 0, err
}

// LastMarker returns the last container name in a ListResult.
func (r ContainerPage) LastMarker() (string, error) {
	names, err := ExtractNames(r)
	if err != nil {
		return "", err
	}
	if len(names) == 0 {
		return "", nil
	}
	return names[len(names)-1], nil
}

// ExtractInfo is a function that takes a ListResult and returns the containers' information.
func ExtractInfo(r pagination.Page) ([]Container, error) {
	var s []Container
	err := (r.(ContainerPage)).ExtractInto(&s)
	return s, err
}

// ExtractNames is a function that takes a ListResult and returns the containers' names.
func ExtractNames(page pagination.Page) ([]string, error) {
	casted := page.(ContainerPage)
	ct := casted.Header.Get("Content-Type")

	switch {
	case strings.HasPrefix(ct, "application/json"):
		parsed, err := ExtractInfo(page)
		if err != nil {
			return nil, err
		}

		names := make([]string, 0, len(parsed))
		for _, container := range parsed {
			names = append(names, container.Name)
		}
		return names, nil
	case strings.HasPrefix(ct, "text/plain"):
		names := make([]string, 0, 50)

		body := string(page.(ContainerPage).Body.([]uint8))
		for _, name := range strings.Split(body, "\n") {
			if len(name) > 0 {
				names = append(names, name)
			}
		}

		return names, nil
	default:
		return nil, fmt.Errorf("Cannot extract names from response with content-type: [%s]", ct)
	}
}

// GetHeader represents the headers returned in the response from a Get request.
type GetHeader struct {
	AcceptRanges     string                  `json:"Accept-Ranges"`
	BytesUsed        int64                   `json:"-"`
	ContentLength    int64                   `json:"-"`
	ContentType      string                  `json:"Content-Type"`
	Date             gophercloud.JSONRFC1123 `json:"Date"`
	ObjectCount      int64                   `json:"-"`
	Read             []string                `json:"-"`
	TransID          string                  `json:"X-Trans-Id"`
	VersionsLocation string                  `json:"X-Versions-Location"`
	Write            []string                `json:"-"`
}

func (h *GetHeader) UnmarshalJSON(b []byte) error {
	type tmp GetHeader
	var getHeader *struct {
		tmp
		BytesUsed     string `json:"X-Container-Bytes-Used"`
		ContentLength string `json:"Content-Length"`
		ObjectCount   string `json:"X-Container-Object-Count"`
		Write         string `json:"X-Container-Write"`
		Read          string `json:"X-Container-Read"`
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

	h.Read = strings.Split(getHeader.Read, ",")
	h.Write = strings.Split(getHeader.Write, ",")

	return nil
}

// GetResult represents the result of a get operation.
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
// and returns the custom metadata associated with the container.
func (r GetResult) ExtractMetadata() (map[string]string, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	metadata := make(map[string]string)
	for k, v := range r.Header {
		if strings.HasPrefix(k, "X-Container-Meta-") {
			key := strings.TrimPrefix(k, "X-Container-Meta-")
			metadata[key] = v[0]
		}
	}
	return metadata, nil
}

// CreateHeader represents the headers returned in the response from a Create request.
type CreateHeader struct {
	ContentLength int64                   `json:"-"`
	ContentType   string                  `json:"Content-Type"`
	Date          gophercloud.JSONRFC1123 `json:"Date"`
	TransID       string                  `json:"X-Trans-Id"`
}

func (h *CreateHeader) UnmarshalJSON(b []byte) error {
	type tmp CreateHeader
	var header *struct {
		tmp
		ContentLength string `json:"Content-Length"`
	}
	err := json.Unmarshal(b, &header)
	if err != nil {
		return err
	}

	*h = CreateHeader(header.tmp)

	switch header.ContentLength {
	case "":
		h.ContentLength = 0
	default:
		h.ContentLength, err = strconv.ParseInt(header.ContentLength, 10, 64)
		if err != nil {
			return err
		}
	}

	return nil
}

// CreateResult represents the result of a create operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type CreateResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Create. To obtain
// a map of headers, call the ExtractHeader method on the CreateResult.
func (r CreateResult) Extract() (*CreateHeader, error) {
	var s *CreateHeader
	err := r.ExtractInto(&s)
	return s, err
}

// UpdateHeader represents the headers returned in the response from a Update request.
type UpdateHeader struct {
	ContentLength int64                   `json:"-"`
	ContentType   string                  `json:"Content-Type"`
	Date          gophercloud.JSONRFC1123 `json:"Date"`
	TransID       string                  `json:"X-Trans-Id"`
}

func (h *UpdateHeader) UnmarshalJSON(b []byte) error {
	type tmp UpdateHeader
	var header *struct {
		tmp
		ContentLength string `json:"Content-Length"`
	}
	err := json.Unmarshal(b, &header)
	if err != nil {
		return err
	}

	*h = UpdateHeader(header.tmp)

	switch header.ContentLength {
	case "":
		h.ContentLength = 0
	default:
		h.ContentLength, err = strconv.ParseInt(header.ContentLength, 10, 64)
		if err != nil {
			return err
		}
	}

	return nil
}

// UpdateResult represents the result of an update operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Update. To obtain
// a map of headers, call the ExtractHeader method on the UpdateResult.
func (r UpdateResult) Extract() (*UpdateHeader, error) {
	var s *UpdateHeader
	err := r.ExtractInto(&s)
	return s, err
}

// DeleteHeader represents the headers returned in the response from a Delete request.
type DeleteHeader struct {
	ContentLength int64                   `json:"-"`
	ContentType   string                  `json:"Content-Type"`
	Date          gophercloud.JSONRFC1123 `json:"Date"`
	TransID       string                  `json:"X-Trans-Id"`
}

func (h *DeleteHeader) UnmarshalJSON(b []byte) error {
	type tmp DeleteHeader
	var header *struct {
		tmp
		ContentLength string `json:"Content-Length"`
	}
	err := json.Unmarshal(b, &header)
	if err != nil {
		return err
	}

	*h = DeleteHeader(header.tmp)

	switch header.ContentLength {
	case "":
		h.ContentLength = 0
	default:
		h.ContentLength, err = strconv.ParseInt(header.ContentLength, 10, 64)
		if err != nil {
			return err
		}
	}

	return nil
}

// DeleteResult represents the result of a delete operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type DeleteResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Delete. To obtain
// a map of headers, call the ExtractHeader method on the DeleteResult.
func (r DeleteResult) Extract() (*DeleteHeader, error) {
	var s *DeleteHeader
	err := r.ExtractInto(&s)
	return s, err
}
