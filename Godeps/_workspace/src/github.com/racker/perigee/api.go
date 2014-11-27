// vim: ts=8 sw=8 noet ai

package perigee

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
)

// The UnexpectedResponseCodeError structure represents a mismatch in understanding between server and client in terms of response codes.
// Most often, this is due to an actual error condition (e.g., getting a 404 for a resource when you expect a 200).
// However, it needn't always be the case (e.g., getting a 204 (No Content) response back when a 200 is expected).
type UnexpectedResponseCodeError struct {
	Url      string
	Expected []int
	Actual   int
	Body     []byte
}

func (err *UnexpectedResponseCodeError) Error() string {
	return fmt.Sprintf("Expected HTTP response code %d when accessing URL(%s); got %d instead with the following body:\n%s", err.Expected, err.Url, err.Actual, string(err.Body))
}

// Request issues an HTTP request, marshaling parameters, and unmarshaling results, as configured in the provided Options parameter.
// The Response structure returned, if any, will include accumulated results recovered from the HTTP server.
// See the Response structure for more details.
func Request(method string, url string, opts Options) (*Response, error) {
	var body io.Reader
	var response Response

	client := opts.CustomClient
	if client == nil {
		client = new(http.Client)
	}

	contentType := opts.ContentType

	body = nil
	if opts.ReqBody != nil {
		if contentType == "" {
			contentType = "application/json"
		}

		if contentType == "application/json" {
			bodyText, err := json.Marshal(opts.ReqBody)
			if err != nil {
				return nil, err
			}
			body = strings.NewReader(string(bodyText))
			if opts.DumpReqJson {
				log.Printf("Making request:\n%#v\n", string(bodyText))
			}
		} else {
			// assume opts.ReqBody implements the correct interface
			body = opts.ReqBody.(io.Reader)
		}
	}

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}

	if contentType != "" {
		req.Header.Add("Content-Type", contentType)
	}

	if opts.ContentLength > 0 {
		req.ContentLength = opts.ContentLength
		req.Header.Add("Content-Length", string(opts.ContentLength))
	}

	if opts.MoreHeaders != nil {
		for k, v := range opts.MoreHeaders {
			req.Header.Add(k, v)
		}
	}

	if accept := req.Header.Get("Accept"); accept == "" {
		accept = opts.Accept
		if accept == "" {
			accept = "application/json"
		}
		req.Header.Add("Accept", accept)
	}

	if opts.SetHeaders != nil {
		err = opts.SetHeaders(req)
		if err != nil {
			return &response, err
		}
	}

	httpResponse, err := client.Do(req)
	if httpResponse != nil {
		response.HttpResponse = *httpResponse
		response.StatusCode = httpResponse.StatusCode
	}

	if err != nil {
		return &response, err
	}
	// This if-statement is legacy code, preserved for backward compatibility.
	if opts.StatusCode != nil {
		*opts.StatusCode = httpResponse.StatusCode
	}

	acceptableResponseCodes := opts.OkCodes
	if len(acceptableResponseCodes) != 0 {
		if not_in(httpResponse.StatusCode, acceptableResponseCodes) {
			b, _ := ioutil.ReadAll(httpResponse.Body)
			httpResponse.Body.Close()
			return &response, &UnexpectedResponseCodeError{
				Url:      url,
				Expected: acceptableResponseCodes,
				Actual:   httpResponse.StatusCode,
				Body:     b,
			}
		}
	}
	if opts.Results != nil {
		defer httpResponse.Body.Close()
		jsonResult, err := ioutil.ReadAll(httpResponse.Body)
		response.JsonResult = jsonResult
		if err != nil {
			return &response, err
		}

		err = json.Unmarshal(jsonResult, opts.Results)
		// This if-statement is legacy code, preserved for backward compatibility.
		if opts.ResponseJson != nil {
			*opts.ResponseJson = jsonResult
		}
	}
	return &response, err
}

// not_in returns false if, and only if, the provided needle is _not_
// in the given set of integers.
func not_in(needle int, haystack []int) bool {
	for _, straw := range haystack {
		if needle == straw {
			return false
		}
	}
	return true
}

// Post makes a POST request against a server using the provided HTTP client.
// The url must be a fully-formed URL string.
// DEPRECATED.  Use Request() instead.
func Post(url string, opts Options) error {
	r, err := Request("POST", url, opts)
	if opts.Response != nil {
		*opts.Response = r
	}
	return err
}

// Get makes a GET request against a server using the provided HTTP client.
// The url must be a fully-formed URL string.
// DEPRECATED.  Use Request() instead.
func Get(url string, opts Options) error {
	r, err := Request("GET", url, opts)
	if opts.Response != nil {
		*opts.Response = r
	}
	return err
}

// Delete makes a DELETE request against a server using the provided HTTP client.
// The url must be a fully-formed URL string.
// DEPRECATED.  Use Request() instead.
func Delete(url string, opts Options) error {
	r, err := Request("DELETE", url, opts)
	if opts.Response != nil {
		*opts.Response = r
	}
	return err
}

// Put makes a PUT request against a server using the provided HTTP client.
// The url must be a fully-formed URL string.
// DEPRECATED.  Use Request() instead.
func Put(url string, opts Options) error {
	r, err := Request("PUT", url, opts)
	if opts.Response != nil {
		*opts.Response = r
	}
	return err
}

// Options describes a set of optional parameters to the various request calls.
//
// The custom client can be used for a variety of purposes beyond selecting encrypted versus unencrypted channels.
// Transports can be defined to provide augmented logging, header manipulation, et. al.
//
// If the ReqBody field is provided, it will be embedded as a JSON object.
// Otherwise, provide nil.
//
// If JSON output is to be expected from the response,
// provide either a pointer to the container structure in Results,
// or a pointer to a nil-initialized pointer variable.
// The latter method will cause the unmarshaller to allocate the container type for you.
// If no response is expected, provide a nil Results value.
//
// The MoreHeaders map, if non-nil or empty, provides a set of headers to add to those
// already present in the request.  At present, only Accepted and Content-Type are set
// by default.
//
// OkCodes provides a set of acceptable, positive responses.
//
// If provided, StatusCode specifies a pointer to an integer, which will receive the
// returned HTTP status code, successful or not.  DEPRECATED; use the Response.StatusCode field instead for new software.
//
// ResponseJson, if specified, provides a means for returning the raw JSON.  This is
// most useful for diagnostics.  DEPRECATED; use the Response.JsonResult field instead for new software.
//
// DumpReqJson, if set to true, will cause the request to appear to stdout for debugging purposes.
// This attribute may be removed at any time in the future; DO NOT use this attribute in production software.
//
// Response, if set, provides a way to communicate the complete set of HTTP response, raw JSON, status code, and
// other useful attributes back to the caller.  Note that the Request() method returns a Response structure as part
// of its public interface; you don't need to set the Response field here to use this structure.  The Response field
// exists primarily for legacy or deprecated functions.
//
// SetHeaders allows the caller to provide code to set any custom headers programmatically.  Typically, this
// facility can invoke, e.g., SetBasicAuth() on the request to easily set up authentication.
// Any error generated will terminate the request and will propegate back to the caller.
type Options struct {
	CustomClient  *http.Client
	ReqBody       interface{}
	Results       interface{}
	MoreHeaders   map[string]string
	OkCodes       []int
	StatusCode    *int    `DEPRECATED`
	DumpReqJson   bool    `UNSUPPORTED`
	ResponseJson  *[]byte `DEPRECATED`
	Response      **Response
	ContentType   string `json:"Content-Type,omitempty"`
	ContentLength int64  `json:"Content-Length,omitempty"`
	Accept        string `json:"Accept,omitempty"`
	SetHeaders    func(r *http.Request) error
}

// Response contains return values from the various request calls.
//
// HttpResponse will return the http response from the request call.
// Note: HttpResponse.Body is always closed and will not be available from this return value.
//
// StatusCode specifies the returned HTTP status code, successful or not.
//
// If Results is specified in the Options:
// - JsonResult will contain the raw return from the request call
//   This is most useful for diagnostics.
// - Result will contain the unmarshalled json either in the Result passed in
//   or the unmarshaller will allocate the container type for you.

type Response struct {
	HttpResponse http.Response
	JsonResult   []byte
	Results      interface{}
	StatusCode   int
}
