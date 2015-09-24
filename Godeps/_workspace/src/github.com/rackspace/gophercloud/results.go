package gophercloud

import (
	"encoding/json"
	"net/http"
	"reflect"

	"github.com/mitchellh/mapstructure"
)

/*
Result is an internal type to be used by individual resource packages, but its
methods will be available on a wide variety of user-facing embedding types.

It acts as a base struct that other Result types, returned from request
functions, can embed for convenience. All Results capture basic information
from the HTTP transaction that was performed, including the response body,
HTTP headers, and any errors that happened.

Generally, each Result type will have an Extract method that can be used to
further interpret the result's payload in a specific context. Extensions or
providers can then provide additional extraction functions to pull out
provider- or extension-specific information as well.
*/
type Result struct {
	// Body is the payload of the HTTP response from the server. In most cases,
	// this will be the deserialized JSON structure.
	Body interface{}

	// Header contains the HTTP header structure from the original response.
	Header http.Header

	// Err is an error that occurred during the operation. It's deferred until
	// extraction to make it easier to chain the Extract call.
	Err error
}

// PrettyPrintJSON creates a string containing the full response body as
// pretty-printed JSON. It's useful for capturing test fixtures and for
// debugging extraction bugs. If you include its output in an issue related to
// a buggy extraction function, we will all love you forever.
func (r Result) PrettyPrintJSON() string {
	pretty, err := json.MarshalIndent(r.Body, "", "  ")
	if err != nil {
		panic(err.Error())
	}
	return string(pretty)
}

// ErrResult is an internal type to be used by individual resource packages, but
// its methods will be available on a wide variety of user-facing embedding
// types.
//
// It represents results that only contain a potential error and
// nothing else. Usually, if the operation executed successfully, the Err field
// will be nil; otherwise it will be stocked with a relevant error. Use the
// ExtractErr method
// to cleanly pull it out.
type ErrResult struct {
	Result
}

// ExtractErr is a function that extracts error information, or nil, from a result.
func (r ErrResult) ExtractErr() error {
	return r.Err
}

/*
HeaderResult is an internal type to be used by individual resource packages, but
its methods will be available on a wide variety of user-facing embedding types.

It represents a result that only contains an error (possibly nil) and an
http.Header. This is used, for example, by the objectstorage packages in
openstack, because most of the operations don't return response bodies, but do
have relevant information in headers.
*/
type HeaderResult struct {
	Result
}

// ExtractHeader will return the http.Header and error from the HeaderResult.
//
//   header, err := objects.Create(client, "my_container", objects.CreateOpts{}).ExtractHeader()
func (hr HeaderResult) ExtractHeader() (http.Header, error) {
	return hr.Header, hr.Err
}

// DecodeHeader is a function that decodes a header (usually of type map[string]interface{}) to
// another type (usually a struct). This function is used by the objectstorage package to give
// users access to response headers without having to query a map. A DecodeHookFunction is used,
// because OpenStack-based clients return header values as arrays (Go slices).
func DecodeHeader(from, to interface{}) error {
	config := &mapstructure.DecoderConfig{
		DecodeHook: func(from, to reflect.Kind, data interface{}) (interface{}, error) {
			if from == reflect.Slice {
				return data.([]string)[0], nil
			}
			return data, nil
		},
		Result:           to,
		WeaklyTypedInput: true,
	}
	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return err
	}
	if err := decoder.Decode(from); err != nil {
		return err
	}
	return nil
}

// RFC3339Milli describes a common time format used by some API responses.
const RFC3339Milli = "2006-01-02T15:04:05.999999Z"

// Time format used in cloud orchestration
const STACK_TIME_FMT = "2006-01-02T15:04:05"

/*
Link is an internal type to be used in packages of collection resources that are
paginated in a certain way.

It's a response substructure common to many paginated collection results that is
used to point to related pages. Usually, the one we care about is the one with
Rel field set to "next".
*/
type Link struct {
	Href string `mapstructure:"href"`
	Rel  string `mapstructure:"rel"`
}

/*
ExtractNextURL is an internal function useful for packages of collection
resources that are paginated in a certain way.

It attempts attempts to extract the "next" URL from slice of Link structs, or
"" if no such URL is present.
*/
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
