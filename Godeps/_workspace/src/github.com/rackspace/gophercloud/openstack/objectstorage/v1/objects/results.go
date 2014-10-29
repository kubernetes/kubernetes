package objects

import (
	"fmt"
	"io"
	"io/ioutil"
	"strings"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Object is a structure that holds information related to a storage object.
type Object struct {
	Bytes        int    `json:"bytes" mapstructure:"bytes"`
	ContentType  string `json:"content_type" mapstructure:"content_type"`
	Hash         string `json:"hash" mapstructure:"hash"`
	LastModified string `json:"last_modified" mapstructure:"last_modified"`
	Name         string `json:"name" mapstructure:"name"`
}

// ObjectPage is a single page of objects that is returned from a call to the
// List function.
type ObjectPage struct {
	pagination.MarkerPageBase
}

// IsEmpty returns true if a ListResult contains no object names.
func (r ObjectPage) IsEmpty() (bool, error) {
	names, err := ExtractNames(r)
	if err != nil {
		return true, err
	}
	return len(names) == 0, nil
}

// LastMarker returns the last object name in a ListResult.
func (r ObjectPage) LastMarker() (string, error) {
	names, err := ExtractNames(r)
	if err != nil {
		return "", err
	}
	if len(names) == 0 {
		return "", nil
	}
	return names[len(names)-1], nil
}

// ExtractInfo is a function that takes a page of objects and returns their full information.
func ExtractInfo(page pagination.Page) ([]Object, error) {
	untyped := page.(ObjectPage).Body.([]interface{})
	results := make([]Object, len(untyped))
	for index, each := range untyped {
		object := each.(map[string]interface{})
		err := mapstructure.Decode(object, &results[index])
		if err != nil {
			return results, err
		}
	}
	return results, nil
}

// ExtractNames is a function that takes a page of objects and returns only their names.
func ExtractNames(page pagination.Page) ([]string, error) {
	casted := page.(ObjectPage)
	ct := casted.Header.Get("Content-Type")
	switch {
	case strings.HasPrefix(ct, "application/json"):
		parsed, err := ExtractInfo(page)
		if err != nil {
			return nil, err
		}

		names := make([]string, 0, len(parsed))
		for _, object := range parsed {
			names = append(names, object.Name)
		}

		return names, nil
	case strings.HasPrefix(ct, "text/plain"):
		names := make([]string, 0, 50)

		body := string(page.(ObjectPage).Body.([]uint8))
		for _, name := range strings.Split(body, "\n") {
			if len(name) > 0 {
				names = append(names, name)
			}
		}

		return names, nil
	case strings.HasPrefix(ct, "text/html"):
		return []string{}, nil
	default:
		return nil, fmt.Errorf("Cannot extract names from response with content-type: [%s]", ct)
	}
}

// DownloadResult is a *http.Response that is returned from a call to the Download function.
type DownloadResult struct {
	gophercloud.HeaderResult
	Body io.ReadCloser
}

// ExtractContent is a function that takes a DownloadResult's io.Reader body
// and reads all available data into a slice of bytes. Please be aware that due
// the nature of io.Reader is forward-only - meaning that it can only be read
// once and not rewound. You can recreate a reader from the output of this
// function by using bytes.NewReader(downloadBytes)
func (dr DownloadResult) ExtractContent() ([]byte, error) {
	if dr.Err != nil {
		return nil, dr.Err
	}
	body, err := ioutil.ReadAll(dr.Body)
	if err != nil {
		return nil, err
	}
	dr.Body.Close()
	return body, nil
}

// GetResult is a *http.Response that is returned from a call to the Get function.
type GetResult struct {
	gophercloud.HeaderResult
}

// ExtractMetadata is a function that takes a GetResult (of type *http.Response)
// and returns the custom metadata associated with the object.
func (gr GetResult) ExtractMetadata() (map[string]string, error) {
	if gr.Err != nil {
		return nil, gr.Err
	}
	metadata := make(map[string]string)
	for k, v := range gr.Header {
		if strings.HasPrefix(k, "X-Object-Meta-") {
			key := strings.TrimPrefix(k, "X-Object-Meta-")
			metadata[key] = v[0]
		}
	}
	return metadata, nil
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	gophercloud.HeaderResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.HeaderResult
}

// CopyResult represents the result of a copy operation.
type CopyResult struct {
	gophercloud.HeaderResult
}
