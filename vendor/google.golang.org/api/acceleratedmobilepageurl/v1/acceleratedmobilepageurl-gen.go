// Package acceleratedmobilepageurl provides access to the Accelerated Mobile Pages (AMP) URL API.
//
// See https://developers.google.com/amp/cache/
//
// Usage example:
//
//   import "google.golang.org/api/acceleratedmobilepageurl/v1"
//   ...
//   acceleratedmobilepageurlService, err := acceleratedmobilepageurl.New(oauthHttpClient)
package acceleratedmobilepageurl // import "google.golang.org/api/acceleratedmobilepageurl/v1"

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

const apiId = "acceleratedmobilepageurl:v1"
const apiName = "acceleratedmobilepageurl"
const apiVersion = "v1"
const basePath = "https://acceleratedmobilepageurl.googleapis.com/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.AmpUrls = NewAmpUrlsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	AmpUrls *AmpUrlsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAmpUrlsService(s *Service) *AmpUrlsService {
	rs := &AmpUrlsService{s: s}
	return rs
}

type AmpUrlsService struct {
	s *Service
}

// AmpUrl: AMP URL response for a requested URL.
type AmpUrl struct {
	// AmpUrl: The AMP URL pointing to the publisher's web server.
	AmpUrl string `json:"ampUrl,omitempty"`

	// CdnAmpUrl: The [AMP Cache
	// URL](/amp/cache/overview#amp-cache-url-format) pointing to
	// the cached document in the Google AMP Cache.
	CdnAmpUrl string `json:"cdnAmpUrl,omitempty"`

	// OriginalUrl: The original non-AMP URL.
	OriginalUrl string `json:"originalUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AmpUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AmpUrl") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AmpUrl) MarshalJSON() ([]byte, error) {
	type noMethod AmpUrl
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AmpUrlError: AMP URL Error resource for a requested URL that couldn't
// be found.
type AmpUrlError struct {
	// ErrorCode: The error code of an API call.
	//
	// Possible values:
	//   "ERROR_CODE_UNSPECIFIED" - Not specified error.
	//   "INPUT_URL_NOT_FOUND" - Indicates the requested URL is not found in
	// the index, possibly because
	// it's unable to be found, not able to be accessed by Googlebot, or
	// some
	// other error.
	//   "NO_AMP_URL" - Indicates no AMP URL has been found that corresponds
	// to the requested
	// URL.
	//   "APPLICATION_ERROR" - Indicates some kind of application error
	// occurred at the server.
	// Client advised to retry.
	//   "URL_IS_VALID_AMP" - DEPRECATED: Indicates the requested URL is a
	// valid AMP URL.  This is a
	// non-error state, should not be relied upon as a sign of success
	// or
	// failure.  It will be removed in future versions of the API.
	//   "URL_IS_INVALID_AMP" - Indicates that an AMP URL has been found
	// that corresponds to the request
	// URL, but it is not valid AMP HTML.
	ErrorCode string `json:"errorCode,omitempty"`

	// ErrorMessage: An optional descriptive error message.
	ErrorMessage string `json:"errorMessage,omitempty"`

	// OriginalUrl: The original non-AMP URL.
	OriginalUrl string `json:"originalUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ErrorCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ErrorCode") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AmpUrlError) MarshalJSON() ([]byte, error) {
	type noMethod AmpUrlError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchGetAmpUrlsRequest: AMP URL request for a batch of URLs.
type BatchGetAmpUrlsRequest struct {
	// LookupStrategy: The lookup_strategy being requested.
	//
	// Possible values:
	//   "FETCH_LIVE_DOC" - FETCH_LIVE_DOC strategy involves live document
	// fetch of URLs not found in
	// the index. Any request URL not found in the index is crawled in
	// realtime
	// to validate if there is a corresponding AMP URL. This strategy has
	// higher
	// coverage but with extra latency introduced by realtime crawling. This
	// is
	// the default strategy. Applications using this strategy should set
	// higher
	// HTTP timeouts of the API calls.
	//   "IN_INDEX_DOC" - IN_INDEX_DOC strategy skips fetching live
	// documents of URL(s) not found
	// in index. For applications which need low latency use of
	// IN_INDEX_DOC
	// strategy is recommended.
	LookupStrategy string `json:"lookupStrategy,omitempty"`

	// Urls: List of URLs to look up for the paired AMP URLs.
	// The URLs are case-sensitive. Up to 50 URLs per lookup
	// (see [Usage Limits](/amp/cache/reference/limits)).
	Urls []string `json:"urls,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LookupStrategy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LookupStrategy") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BatchGetAmpUrlsRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchGetAmpUrlsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchGetAmpUrlsResponse: Batch AMP URL response.
type BatchGetAmpUrlsResponse struct {
	// AmpUrls: For each URL in BatchAmpUrlsRequest, the URL response. The
	// response might
	// not be in the same order as URLs in the batch request.
	// If BatchAmpUrlsRequest contains duplicate URLs, AmpUrl is
	// generated
	// only once.
	AmpUrls []*AmpUrl `json:"ampUrls,omitempty"`

	// UrlErrors: The errors for requested URLs that have no AMP URL.
	UrlErrors []*AmpUrlError `json:"urlErrors,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AmpUrls") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AmpUrls") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchGetAmpUrlsResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchGetAmpUrlsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "acceleratedmobilepageurl.ampUrls.batchGet":

type AmpUrlsBatchGetCall struct {
	s                      *Service
	batchgetampurlsrequest *BatchGetAmpUrlsRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// BatchGet: Returns AMP URL(s) and equivalent
// [AMP Cache URL(s)](/amp/cache/overview#amp-cache-url-format).
func (r *AmpUrlsService) BatchGet(batchgetampurlsrequest *BatchGetAmpUrlsRequest) *AmpUrlsBatchGetCall {
	c := &AmpUrlsBatchGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.batchgetampurlsrequest = batchgetampurlsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AmpUrlsBatchGetCall) Fields(s ...googleapi.Field) *AmpUrlsBatchGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AmpUrlsBatchGetCall) Context(ctx context.Context) *AmpUrlsBatchGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AmpUrlsBatchGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AmpUrlsBatchGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchgetampurlsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/ampUrls:batchGet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "acceleratedmobilepageurl.ampUrls.batchGet" call.
// Exactly one of *BatchGetAmpUrlsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *BatchGetAmpUrlsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AmpUrlsBatchGetCall) Do(opts ...googleapi.CallOption) (*BatchGetAmpUrlsResponse, error) {
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
	ret := &BatchGetAmpUrlsResponse{
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
	//   "description": "Returns AMP URL(s) and equivalent\n[AMP Cache URL(s)](/amp/cache/overview#amp-cache-url-format).",
	//   "flatPath": "v1/ampUrls:batchGet",
	//   "httpMethod": "POST",
	//   "id": "acceleratedmobilepageurl.ampUrls.batchGet",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/ampUrls:batchGet",
	//   "request": {
	//     "$ref": "BatchGetAmpUrlsRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchGetAmpUrlsResponse"
	//   }
	// }

}
