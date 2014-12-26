package containers

import (
	"fmt"
	"strings"

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

// GetResult represents the result of a get operation.
type GetResult struct {
	gophercloud.HeaderResult
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

// CreateResult represents the result of a create operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type CreateResult struct {
	gophercloud.HeaderResult
}

// UpdateResult represents the result of an update operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type UpdateResult struct {
	gophercloud.HeaderResult
}

// DeleteResult represents the result of a delete operation. To extract the
// the headers from the HTTP response, you can invoke the 'ExtractHeader'
// method on the result struct.
type DeleteResult struct {
	gophercloud.HeaderResult
}
