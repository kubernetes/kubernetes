// Package webfonts provides access to the Google Fonts Developer API.
//
// See https://developers.google.com/fonts/docs/developer_api
//
// Usage example:
//
//   import "google.golang.org/api/webfonts/v1"
//   ...
//   webfontsService, err := webfonts.New(oauthHttpClient)
package webfonts // import "google.golang.org/api/webfonts/v1"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	context "golang.org/x/net/context"
	ctxhttp "golang.org/x/net/context/ctxhttp"
	gensupport "google.golang.org/api/gensupport"
	googleapi "google.golang.org/api/googleapi"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = gensupport.MarshalJSON
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Canceled
var _ = ctxhttp.Do

const apiId = "webfonts:v1"
const apiName = "webfonts"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/webfonts/v1/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Webfonts = NewWebfontsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Webfonts *WebfontsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewWebfontsService(s *Service) *WebfontsService {
	rs := &WebfontsService{s: s}
	return rs
}

type WebfontsService struct {
	s *Service
}

type Webfont struct {
	// Category: The category of the font.
	Category string `json:"category,omitempty"`

	// Family: The name of the font.
	Family string `json:"family,omitempty"`

	// Files: The font files (with all supported scripts) for each one of
	// the available variants, as a key : value map.
	Files map[string]string `json:"files,omitempty"`

	// Kind: This kind represents a webfont object in the webfonts service.
	Kind string `json:"kind,omitempty"`

	// LastModified: The date (format "yyyy-MM-dd") the font was modified
	// for the last time.
	LastModified string `json:"lastModified,omitempty"`

	// Subsets: The scripts supported by the font.
	Subsets []string `json:"subsets,omitempty"`

	// Variants: The available variants for the font.
	Variants []string `json:"variants,omitempty"`

	// Version: The font version.
	Version string `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Category") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Category") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Webfont) MarshalJSON() ([]byte, error) {
	type noMethod Webfont
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type WebfontList struct {
	// Items: The list of fonts currently served by the Google Fonts API.
	Items []*Webfont `json:"items,omitempty"`

	// Kind: This kind represents a list of webfont objects in the webfonts
	// service.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WebfontList) MarshalJSON() ([]byte, error) {
	type noMethod WebfontList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "webfonts.webfonts.list":

type WebfontsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the list of fonts currently served by the Google
// Fonts Developer API
func (r *WebfontsService) List() *WebfontsListCall {
	c := &WebfontsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Sort sets the optional parameter "sort": Enables sorting of the list
//
// Possible values:
//   "alpha" - Sort alphabetically
//   "date" - Sort by date added
//   "popularity" - Sort by popularity
//   "style" - Sort by number of styles
//   "trending" - Sort by trending
func (c *WebfontsListCall) Sort(sort string) *WebfontsListCall {
	c.urlParams_.Set("sort", sort)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *WebfontsListCall) Fields(s ...googleapi.Field) *WebfontsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *WebfontsListCall) IfNoneMatch(entityTag string) *WebfontsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *WebfontsListCall) Context(ctx context.Context) *WebfontsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *WebfontsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *WebfontsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "webfonts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "webfonts.webfonts.list" call.
// Exactly one of *WebfontList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *WebfontList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *WebfontsListCall) Do(opts ...googleapi.CallOption) (*WebfontList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &WebfontList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of fonts currently served by the Google Fonts Developer API",
	//   "httpMethod": "GET",
	//   "id": "webfonts.webfonts.list",
	//   "parameters": {
	//     "sort": {
	//       "description": "Enables sorting of the list",
	//       "enum": [
	//         "alpha",
	//         "date",
	//         "popularity",
	//         "style",
	//         "trending"
	//       ],
	//       "enumDescriptions": [
	//         "Sort alphabetically",
	//         "Sort by date added",
	//         "Sort by popularity",
	//         "Sort by number of styles",
	//         "Sort by trending"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "webfonts",
	//   "response": {
	//     "$ref": "WebfontList"
	//   }
	// }

}
