package accounts

import (
	"strings"

	"github.com/rackspace/gophercloud"
)

// GetResult is returned from a call to the Get function.
type GetResult struct {
	gophercloud.HeaderResult
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

// UpdateResult is returned from a call to the Update function.
type UpdateResult struct {
	gophercloud.HeaderResult
}
