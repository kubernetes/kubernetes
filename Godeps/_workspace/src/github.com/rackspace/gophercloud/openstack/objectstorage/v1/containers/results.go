package containers

import (
	"fmt"
	"strings"
	"time"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Container represents a container resource.
type Container struct {
	// The total number of bytes stored in the container.
	Bytes int `json:"bytes" mapstructure:"bytes"`

	// The total number of objects stored in the container.
	Count int `json:"count" mapstructure:"count"`

	// The name of the container.
	Name string `json:"name" mapstructure:"name"`
}

// ContainerPage is the page returned by a pager when traversing over a
// collection of containers.
type ContainerPage struct {
	pagination.MarkerPageBase
}

// IsEmpty returns true if a ListResult contains no container names.
func (r ContainerPage) IsEmpty() (bool, error) {
	names, err := ExtractNames(r)
	if err != nil {
		return true, err
	}
	return len(names) == 0, nil
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
func ExtractInfo(page pagination.Page) ([]Container, error) {
	untyped := page.(ContainerPage).Body.([]interface{})
	results := make([]Container, len(untyped))
	for index, each := range untyped {
		container := each.(map[string]interface{})
		err := mapstructure.Decode(container, &results[index])
		if err != nil {
			return results, err
		}
	}
	return results, nil
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
	AcceptRanges     string    `mapstructure:"Accept-Ranges"`
	BytesUsed        int64     `mapstructure:"X-Account-Bytes-Used"`
	ContentLength    int64     `mapstructure:"Content-Length"`
	ContentType      string    `mapstructure:"Content-Type"`
	Date             time.Time `mapstructure:"-"`
	ObjectCount      int64     `mapstructure:"X-Container-Object-Count"`
	Read             string    `mapstructure:"X-Container-Read"`
	TransID          string    `mapstructure:"X-Trans-Id"`
	VersionsLocation string    `mapstructure:"X-Versions-Location"`
	Write            string    `mapstructure:"X-Container-Write"`
}

// GetResult represents the result of a get operation.
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
// and returns the custom metadata associated with the container.
func (gr GetResult) ExtractMetadata() (map[string]string, error) {
	if gr.Err != nil {
		return nil, gr.Err
	}
	metadata := make(map[string]string)
	for k, v := range gr.Header {
		if strings.HasPrefix(k, "X-Container-Meta-") {
			key := strings.TrimPrefix(k, "X-Container-Meta-")
			metadata[key] = v[0]
		}
	}
	return metadata, nil
}

// CreateHeader represents the headers returned in the response from a Create request.
type CreateHeader struct {
	ContentLength int64     `mapstructure:"Content-Length"`
	ContentType   string    `mapstructure:"Content-Type"`
	Date          time.Time `mapstructure:"-"`
	TransID       string    `mapstructure:"X-Trans-Id"`
}

// CreateResult represents the result of a create operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type CreateResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Create. To obtain
// a map of headers, call the ExtractHeader method on the CreateResult.
func (cr CreateResult) Extract() (CreateHeader, error) {
	var ch CreateHeader
	if cr.Err != nil {
		return ch, cr.Err
	}

	if err := gophercloud.DecodeHeader(cr.Header, &ch); err != nil {
		return ch, err
	}

	if date, ok := cr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, cr.Header["Date"][0])
		if err != nil {
			return ch, err
		}
		ch.Date = t
	}

	return ch, nil
}

// UpdateHeader represents the headers returned in the response from a Update request.
type UpdateHeader struct {
	ContentLength int64     `mapstructure:"Content-Length"`
	ContentType   string    `mapstructure:"Content-Type"`
	Date          time.Time `mapstructure:"-"`
	TransID       string    `mapstructure:"X-Trans-Id"`
}

// UpdateResult represents the result of an update operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Update. To obtain
// a map of headers, call the ExtractHeader method on the UpdateResult.
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

// DeleteHeader represents the headers returned in the response from a Delete request.
type DeleteHeader struct {
	ContentLength int64     `mapstructure:"Content-Length"`
	ContentType   string    `mapstructure:"Content-Type"`
	Date          time.Time `mapstructure:"-"`
	TransID       string    `mapstructure:"X-Trans-Id"`
}

// DeleteResult represents the result of a delete operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type DeleteResult struct {
	gophercloud.HeaderResult
}

// Extract will return a struct of headers returned from a call to Delete. To obtain
// a map of headers, call the ExtractHeader method on the DeleteResult.
func (dr DeleteResult) Extract() (DeleteHeader, error) {
	var dh DeleteHeader
	if dr.Err != nil {
		return dh, dr.Err
	}

	if err := gophercloud.DecodeHeader(dr.Header, &dh); err != nil {
		return dh, err
	}

	if date, ok := dr.Header["Date"]; ok && len(date) > 0 {
		t, err := time.Parse(time.RFC1123, dr.Header["Date"][0])
		if err != nil {
			return dh, err
		}
		dh.Date = t
	}

	return dh, nil
}
