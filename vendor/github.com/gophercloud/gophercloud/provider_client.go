package gophercloud

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
)

// DefaultUserAgent is the default User-Agent string set in the request header.
const DefaultUserAgent = "gophercloud/2.0.0"

// UserAgent represents a User-Agent header.
type UserAgent struct {
	// prepend is the slice of User-Agent strings to prepend to DefaultUserAgent.
	// All the strings to prepend are accumulated and prepended in the Join method.
	prepend []string
}

// Prepend prepends a user-defined string to the default User-Agent string. Users
// may pass in one or more strings to prepend.
func (ua *UserAgent) Prepend(s ...string) {
	ua.prepend = append(s, ua.prepend...)
}

// Join concatenates all the user-defined User-Agend strings with the default
// Gophercloud User-Agent string.
func (ua *UserAgent) Join() string {
	uaSlice := append(ua.prepend, DefaultUserAgent)
	return strings.Join(uaSlice, " ")
}

// ProviderClient stores details that are required to interact with any
// services within a specific provider's API.
//
// Generally, you acquire a ProviderClient by calling the NewClient method in
// the appropriate provider's child package, providing whatever authentication
// credentials are required.
type ProviderClient struct {
	// IdentityBase is the base URL used for a particular provider's identity
	// service - it will be used when issuing authenticatation requests. It
	// should point to the root resource of the identity service, not a specific
	// identity version.
	IdentityBase string

	// IdentityEndpoint is the identity endpoint. This may be a specific version
	// of the identity service. If this is the case, this endpoint is used rather
	// than querying versions first.
	IdentityEndpoint string

	// TokenID is the ID of the most recently issued valid token.
	TokenID string

	// EndpointLocator describes how this provider discovers the endpoints for
	// its constituent services.
	EndpointLocator EndpointLocator

	// HTTPClient allows users to interject arbitrary http, https, or other transit behaviors.
	HTTPClient http.Client

	// UserAgent represents the User-Agent header in the HTTP request.
	UserAgent UserAgent

	// ReauthFunc is the function used to re-authenticate the user if the request
	// fails with a 401 HTTP response code. This a needed because there may be multiple
	// authentication functions for different Identity service versions.
	ReauthFunc func() error

	Debug bool
}

// AuthenticatedHeaders returns a map of HTTP headers that are common for all
// authenticated service requests.
func (client *ProviderClient) AuthenticatedHeaders() map[string]string {
	if client.TokenID == "" {
		return map[string]string{}
	}
	return map[string]string{"X-Auth-Token": client.TokenID}
}

// RequestOpts customizes the behavior of the provider.Request() method.
type RequestOpts struct {
	// JSONBody, if provided, will be encoded as JSON and used as the body of the HTTP request. The
	// content type of the request will default to "application/json" unless overridden by MoreHeaders.
	// It's an error to specify both a JSONBody and a RawBody.
	JSONBody interface{}
	// RawBody contains an io.Reader that will be consumed by the request directly. No content-type
	// will be set unless one is provided explicitly by MoreHeaders.
	RawBody io.Reader
	// JSONResponse, if provided, will be populated with the contents of the response body parsed as
	// JSON.
	JSONResponse interface{}
	// OkCodes contains a list of numeric HTTP status codes that should be interpreted as success. If
	// the response has a different code, an error will be returned.
	OkCodes []int
	// MoreHeaders specifies additional HTTP headers to be provide on the request. If a header is
	// provided with a blank value (""), that header will be *omitted* instead: use this to suppress
	// the default Accept header or an inferred Content-Type, for example.
	MoreHeaders map[string]string
	// ErrorContext specifies the resource error type to return if an error is encountered.
	// This lets resources override default error messages based on the response status code.
	ErrorContext error
}

var applicationJSON = "application/json"

// Request performs an HTTP request using the ProviderClient's current HTTPClient. An authentication
// header will automatically be provided.
func (client *ProviderClient) Request(method, url string, options *RequestOpts) (*http.Response, error) {
	var body io.Reader
	var contentType *string

	// Derive the content body by either encoding an arbitrary object as JSON, or by taking a provided
	// io.ReadSeeker as-is. Default the content-type to application/json.
	if options.JSONBody != nil {
		if options.RawBody != nil {
			panic("Please provide only one of JSONBody or RawBody to gophercloud.Request().")
		}

		rendered, err := json.Marshal(options.JSONBody)
		if err != nil {
			return nil, err
		}

		body = bytes.NewReader(rendered)
		contentType = &applicationJSON
	}

	if options.RawBody != nil {
		body = options.RawBody
	}

	// Construct the http.Request.
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}

	// Populate the request headers. Apply options.MoreHeaders last, to give the caller the chance to
	// modify or omit any header.
	if contentType != nil {
		req.Header.Set("Content-Type", *contentType)
	}
	req.Header.Set("Accept", applicationJSON)

	for k, v := range client.AuthenticatedHeaders() {
		req.Header.Add(k, v)
	}

	// Set the User-Agent header
	req.Header.Set("User-Agent", client.UserAgent.Join())

	if options.MoreHeaders != nil {
		for k, v := range options.MoreHeaders {
			if v != "" {
				req.Header.Set(k, v)
			} else {
				req.Header.Del(k)
			}
		}
	}

	// Set connection parameter to close the connection immediately when we've got the response
	req.Close = true

	// Issue the request.
	resp, err := client.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}

	// Allow default OkCodes if none explicitly set
	if options.OkCodes == nil {
		options.OkCodes = defaultOkCodes(method)
	}

	// Validate the HTTP response status.
	var ok bool
	for _, code := range options.OkCodes {
		if resp.StatusCode == code {
			ok = true
			break
		}
	}

	if !ok {
		body, _ := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		//pc := make([]uintptr, 1)
		//runtime.Callers(2, pc)
		//f := runtime.FuncForPC(pc[0])
		respErr := ErrUnexpectedResponseCode{
			URL:      url,
			Method:   method,
			Expected: options.OkCodes,
			Actual:   resp.StatusCode,
			Body:     body,
		}
		//respErr.Function = "gophercloud.ProviderClient.Request"

		errType := options.ErrorContext
		switch resp.StatusCode {
		case http.StatusBadRequest:
			err = ErrDefault400{respErr}
			if error400er, ok := errType.(Err400er); ok {
				err = error400er.Error400(respErr)
			}
		case http.StatusUnauthorized:
			if client.ReauthFunc != nil {
				err = client.ReauthFunc()
				if err != nil {
					e := &ErrUnableToReauthenticate{}
					e.ErrOriginal = respErr
					return nil, e
				}
				if options.RawBody != nil {
					if seeker, ok := options.RawBody.(io.Seeker); ok {
						seeker.Seek(0, 0)
					}
				}
				resp, err = client.Request(method, url, options)
				if err != nil {
					switch err.(type) {
					case *ErrUnexpectedResponseCode:
						e := &ErrErrorAfterReauthentication{}
						e.ErrOriginal = err.(*ErrUnexpectedResponseCode)
						return nil, e
					default:
						e := &ErrErrorAfterReauthentication{}
						e.ErrOriginal = err
						return nil, e
					}
				}
				return resp, nil
			}
			err = ErrDefault401{respErr}
			if error401er, ok := errType.(Err401er); ok {
				err = error401er.Error401(respErr)
			}
		case http.StatusNotFound:
			err = ErrDefault404{respErr}
			if error404er, ok := errType.(Err404er); ok {
				err = error404er.Error404(respErr)
			}
		case http.StatusMethodNotAllowed:
			err = ErrDefault405{respErr}
			if error405er, ok := errType.(Err405er); ok {
				err = error405er.Error405(respErr)
			}
		case http.StatusRequestTimeout:
			err = ErrDefault408{respErr}
			if error408er, ok := errType.(Err408er); ok {
				err = error408er.Error408(respErr)
			}
		case 429:
			err = ErrDefault429{respErr}
			if error429er, ok := errType.(Err429er); ok {
				err = error429er.Error429(respErr)
			}
		case http.StatusInternalServerError:
			err = ErrDefault500{respErr}
			if error500er, ok := errType.(Err500er); ok {
				err = error500er.Error500(respErr)
			}
		case http.StatusServiceUnavailable:
			err = ErrDefault503{respErr}
			if error503er, ok := errType.(Err503er); ok {
				err = error503er.Error503(respErr)
			}
		}

		if err == nil {
			err = respErr
		}

		return resp, err
	}

	// Parse the response body as JSON, if requested to do so.
	if options.JSONResponse != nil {
		defer resp.Body.Close()
		if err := json.NewDecoder(resp.Body).Decode(options.JSONResponse); err != nil {
			return nil, err
		}
	}

	return resp, nil
}

func defaultOkCodes(method string) []int {
	switch {
	case method == "GET":
		return []int{200}
	case method == "POST":
		return []int{201, 202}
	case method == "PUT":
		return []int{201, 202}
	case method == "PATCH":
		return []int{200, 204}
	case method == "DELETE":
		return []int{202, 204}
	}

	return []int{}
}
