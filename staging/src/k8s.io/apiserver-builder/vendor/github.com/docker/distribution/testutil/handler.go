package testutil

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

// RequestResponseMap is an ordered mapping from Requests to Responses
type RequestResponseMap []RequestResponseMapping

// RequestResponseMapping defines a Response to be sent in response to a given
// Request
type RequestResponseMapping struct {
	Request  Request
	Response Response
}

// Request is a simplified http.Request object
type Request struct {
	// Method is the http method of the request, for example GET
	Method string

	// Route is the http route of this request
	Route string

	// QueryParams are the query parameters of this request
	QueryParams map[string][]string

	// Body is the byte contents of the http request
	Body []byte

	// Headers are the header for this request
	Headers http.Header
}

func (r Request) String() string {
	queryString := ""
	if len(r.QueryParams) > 0 {
		keys := make([]string, 0, len(r.QueryParams))
		queryParts := make([]string, 0, len(r.QueryParams))
		for k := range r.QueryParams {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			for _, val := range r.QueryParams[k] {
				queryParts = append(queryParts, fmt.Sprintf("%s=%s", k, url.QueryEscape(val)))
			}
		}
		queryString = "?" + strings.Join(queryParts, "&")
	}
	var headers []string
	if len(r.Headers) > 0 {
		var headerKeys []string
		for k := range r.Headers {
			headerKeys = append(headerKeys, k)
		}
		sort.Strings(headerKeys)

		for _, k := range headerKeys {
			for _, val := range r.Headers[k] {
				headers = append(headers, fmt.Sprintf("%s:%s", k, val))
			}
		}

	}
	return fmt.Sprintf("%s %s%s\n%s\n%s", r.Method, r.Route, queryString, headers, r.Body)
}

// Response is a simplified http.Response object
type Response struct {
	// Statuscode is the http status code of the Response
	StatusCode int

	// Headers are the http headers of this Response
	Headers http.Header

	// Body is the response body
	Body []byte
}

// testHandler is an http.Handler with a defined mapping from Request to an
// ordered list of Response objects
type testHandler struct {
	responseMap map[string][]Response
}

// NewHandler returns a new test handler that responds to defined requests
// with specified responses
// Each time a Request is received, the next Response is returned in the
// mapping, until no Responses are defined, at which point a 404 is sent back
func NewHandler(requestResponseMap RequestResponseMap) http.Handler {
	responseMap := make(map[string][]Response)
	for _, mapping := range requestResponseMap {
		responses, ok := responseMap[mapping.Request.String()]
		if ok {
			responseMap[mapping.Request.String()] = append(responses, mapping.Response)
		} else {
			responseMap[mapping.Request.String()] = []Response{mapping.Response}
		}
	}
	return &testHandler{responseMap: responseMap}
}

func (app *testHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	requestBody, _ := ioutil.ReadAll(r.Body)
	request := Request{
		Method:      r.Method,
		Route:       r.URL.Path,
		QueryParams: r.URL.Query(),
		Body:        requestBody,
		Headers:     make(map[string][]string),
	}

	// Add headers of interest here
	for k, v := range r.Header {
		if k == "If-None-Match" {
			request.Headers[k] = v
		}
	}

	responses, ok := app.responseMap[request.String()]

	if !ok || len(responses) == 0 {
		http.NotFound(w, r)
		return
	}

	response := responses[0]
	app.responseMap[request.String()] = responses[1:]

	responseHeader := w.Header()
	for k, v := range response.Headers {
		responseHeader[k] = v
	}

	w.WriteHeader(response.StatusCode)

	io.Copy(w, bytes.NewReader(response.Body))
}
