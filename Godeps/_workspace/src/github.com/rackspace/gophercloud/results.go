package gophercloud

import (
	"encoding/json"
	"net/http"
)

// Result acts as a base struct that other results can embed.
type Result struct {
	// Body is the payload of the HTTP response from the server. In most cases, this will be the
	// deserialized JSON structure.
	Body interface{}

	// Header contains the HTTP header structure from the original response.
	Header http.Header

	// Err is an error that occurred during the operation. It's deferred until extraction to make
	// it easier to chain operations.
	Err error
}

// PrettyPrintJSON creates a string containing the full response body as pretty-printed JSON.
func (r Result) PrettyPrintJSON() string {
	pretty, err := json.MarshalIndent(r.Body, "", "  ")
	if err != nil {
		panic(err.Error())
	}
	return string(pretty)
}

// ErrResult represents results that only contain a potential error and
// nothing else. Usually if the operation executed successfully, the Err field
// will be nil; otherwise it will be stocked with a relevant error.
type ErrResult struct {
	Result
}

// ExtractErr is a function that extracts error information from a result.
func (r ErrResult) ExtractErr() error {
	return r.Err
}

// HeaderResult represents a result that only contains an `error` (possibly nil)
// and an http.Header. This is used, for example, by the `objectstorage` packages
// in `openstack`, because most of the operations don't return response bodies.
type HeaderResult struct {
	Result
}

// ExtractHeader will return the http.Header and error from the HeaderResult.
// Usage: header, err := objects.Create(client, "my_container", objects.CreateOpts{}).ExtractHeader()
func (hr HeaderResult) ExtractHeader() (http.Header, error) {
	return hr.Header, hr.Err
}

// RFC3339Milli describes a time format used by API responses.
const RFC3339Milli = "2006-01-02T15:04:05.999999Z"

// Link represents a structure that enables paginated collections how to
// traverse backward or forward. The "Rel" field is usually either "next".
type Link struct {
	Href string `mapstructure:"href"`
	Rel  string `mapstructure:"rel"`
}

// ExtractNextURL attempts to extract the next URL from a JSON structure. It
// follows the common convention of nesting back and next URLs in a "links"
// JSON array.
func ExtractNextURL(links []Link) (string, error) {
	var url string

	for _, l := range links {
		if l.Rel == "next" {
			url = l.Href
		}
	}

	if url == "" {
		return "", nil
	}

	return url, nil
}
