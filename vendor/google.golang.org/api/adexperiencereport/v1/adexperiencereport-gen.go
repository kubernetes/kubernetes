// Package adexperiencereport provides access to the Google Ad Experience Report API.
//
// See https://developers.google.com/ad-experience-report/
//
// Usage example:
//
//   import "google.golang.org/api/adexperiencereport/v1"
//   ...
//   adexperiencereportService, err := adexperiencereport.New(oauthHttpClient)
package adexperiencereport // import "google.golang.org/api/adexperiencereport/v1"

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

const apiId = "adexperiencereport:v1"
const apiName = "adexperiencereport"
const apiVersion = "v1"
const basePath = "https://adexperiencereport.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// Test scope for access to the Zoo service
	XapiZooScope = "https://www.googleapis.com/auth/xapi.zoo"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Sites = NewSitesService(s)
	s.ViolatingSites = NewViolatingSitesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Sites *SitesService

	ViolatingSites *ViolatingSitesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewSitesService(s *Service) *SitesService {
	rs := &SitesService{s: s}
	return rs
}

type SitesService struct {
	s *Service
}

func NewViolatingSitesService(s *Service) *ViolatingSitesService {
	rs := &ViolatingSitesService{s: s}
	return rs
}

type ViolatingSitesService struct {
	s *Service
}

// PlatformSummary: Summary of the ad experience rating of a site for a
// specific platform.
type PlatformSummary struct {
	// AbusiveStatus: The status of the site reviewed for abusive ads.
	//
	// Possible values:
	//   "UNKNOWN" - Not reviewed.
	//   "PASSING" - Passing.
	//   "FAILING" - Failing.
	AbusiveStatus string `json:"abusiveStatus,omitempty"`

	// BetterAdsStatus: The status of the site reviewed for the Better Ads
	// Standards.
	//
	// Possible values:
	//   "UNKNOWN" - Not reviewed.
	//   "PASSING" - Passing.
	//   "WARNING" - Warning.
	//   "FAILING" - Failing.
	BetterAdsStatus string `json:"betterAdsStatus,omitempty"`

	// EnforcementTime: The date on which ad filtering begins.
	EnforcementTime string `json:"enforcementTime,omitempty"`

	// FilterStatus: The ad filtering status of the site.
	//
	// Possible values:
	//   "UNKNOWN" - N/A.
	//   "ON" - Ad filtering is on.
	//   "OFF" - Ad filtering is off.
	//   "PAUSED" - Ad filtering is paused.
	//   "PENDING" - Ad filtering is pending.
	FilterStatus string `json:"filterStatus,omitempty"`

	// LastChangeTime: The last time that the site changed status.
	LastChangeTime string `json:"lastChangeTime,omitempty"`

	// Region: The assigned regions for the site and platform.
	//
	// Possible values:
	//   "REGION_UNKNOWN" - Ad standard not yet defined for your region.
	//   "REGION_A" - Region A.
	//   "REGION_B" - Region B.
	Region []string `json:"region,omitempty"`

	// ReportUrl: A link that leads to a full ad experience report.
	ReportUrl string `json:"reportUrl,omitempty"`

	// UnderReview: Whether the site is currently under review.
	UnderReview bool `json:"underReview,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AbusiveStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AbusiveStatus") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PlatformSummary) MarshalJSON() ([]byte, error) {
	type noMethod PlatformSummary
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SiteSummaryResponse: Response message for GetSiteSummary.
type SiteSummaryResponse struct {
	// DesktopSummary: Summary for the desktop review of the site.
	DesktopSummary *PlatformSummary `json:"desktopSummary,omitempty"`

	// MobileSummary: Summary for the mobile review of the site.
	MobileSummary *PlatformSummary `json:"mobileSummary,omitempty"`

	// ReviewedSite: The name of the site reviewed.
	ReviewedSite string `json:"reviewedSite,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DesktopSummary") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DesktopSummary") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SiteSummaryResponse) MarshalJSON() ([]byte, error) {
	type noMethod SiteSummaryResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ViolatingSitesResponse: Response message for ListViolatingSites.
type ViolatingSitesResponse struct {
	// ViolatingSites: A list of summaries of violating sites.
	ViolatingSites []*SiteSummaryResponse `json:"violatingSites,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ViolatingSites") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ViolatingSites") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ViolatingSitesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ViolatingSitesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "adexperiencereport.sites.get":

type SitesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a summary of the ad experience rating of a site.
func (r *SitesService) Get(name string) *SitesGetCall {
	c := &SitesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SitesGetCall) Fields(s ...googleapi.Field) *SitesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SitesGetCall) IfNoneMatch(entityTag string) *SitesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SitesGetCall) Context(ctx context.Context) *SitesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SitesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SitesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexperiencereport.sites.get" call.
// Exactly one of *SiteSummaryResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SiteSummaryResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SitesGetCall) Do(opts ...googleapi.CallOption) (*SiteSummaryResponse, error) {
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
	ret := &SiteSummaryResponse{
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
	//   "description": "Gets a summary of the ad experience rating of a site.",
	//   "flatPath": "v1/sites/{sitesId}",
	//   "httpMethod": "GET",
	//   "id": "adexperiencereport.sites.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The required site name. It should be the site property whose ad experiences\nmay have been reviewed, and it should be URL-encoded. For example,\nsites/https%3A%2F%2Fwww.google.com. The server will return an error of\nBAD_REQUEST if this field is not filled in. Note that if the site property\nis not yet verified in Search Console, the reportUrl field returned by the\nAPI will lead to the verification page, prompting the user to go through\nthat process before they can gain access to the Ad Experience Report.",
	//       "location": "path",
	//       "pattern": "^sites/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "SiteSummaryResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/xapi.zoo"
	//   ]
	// }

}

// method id "adexperiencereport.violatingSites.list":

type ViolatingSitesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists sites with Ad Experience Report statuses of "Failing" or
// "Warning".
func (r *ViolatingSitesService) List() *ViolatingSitesListCall {
	c := &ViolatingSitesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ViolatingSitesListCall) Fields(s ...googleapi.Field) *ViolatingSitesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ViolatingSitesListCall) IfNoneMatch(entityTag string) *ViolatingSitesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ViolatingSitesListCall) Context(ctx context.Context) *ViolatingSitesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ViolatingSitesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ViolatingSitesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/violatingSites")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexperiencereport.violatingSites.list" call.
// Exactly one of *ViolatingSitesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ViolatingSitesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ViolatingSitesListCall) Do(opts ...googleapi.CallOption) (*ViolatingSitesResponse, error) {
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
	ret := &ViolatingSitesResponse{
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
	//   "description": "Lists sites with Ad Experience Report statuses of \"Failing\" or \"Warning\".",
	//   "flatPath": "v1/violatingSites",
	//   "httpMethod": "GET",
	//   "id": "adexperiencereport.violatingSites.list",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/violatingSites",
	//   "response": {
	//     "$ref": "ViolatingSitesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/xapi.zoo"
	//   ]
	// }

}
