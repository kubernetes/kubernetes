// Package adexchangeseller provides access to the Ad Exchange Seller API.
//
// See https://developers.google.com/ad-exchange/seller-rest/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/adexchangeseller/v1"
//   ...
//   adexchangesellerService, err := adexchangeseller.New(oauthHttpClient)
package adexchangeseller

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
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
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "adexchangeseller:v1"
const apiName = "adexchangeseller"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/adexchangeseller/v1/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Ad Exchange data
	AdexchangeSellerScope = "https://www.googleapis.com/auth/adexchange.seller"

	// View your Ad Exchange data
	AdexchangeSellerReadonlyScope = "https://www.googleapis.com/auth/adexchange.seller.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Adclients = NewAdclientsService(s)
	s.Adunits = NewAdunitsService(s)
	s.Customchannels = NewCustomchannelsService(s)
	s.Reports = NewReportsService(s)
	s.Urlchannels = NewUrlchannelsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Adclients *AdclientsService

	Adunits *AdunitsService

	Customchannels *CustomchannelsService

	Reports *ReportsService

	Urlchannels *UrlchannelsService
}

func NewAdclientsService(s *Service) *AdclientsService {
	rs := &AdclientsService{s: s}
	return rs
}

type AdclientsService struct {
	s *Service
}

func NewAdunitsService(s *Service) *AdunitsService {
	rs := &AdunitsService{s: s}
	rs.Customchannels = NewAdunitsCustomchannelsService(s)
	return rs
}

type AdunitsService struct {
	s *Service

	Customchannels *AdunitsCustomchannelsService
}

func NewAdunitsCustomchannelsService(s *Service) *AdunitsCustomchannelsService {
	rs := &AdunitsCustomchannelsService{s: s}
	return rs
}

type AdunitsCustomchannelsService struct {
	s *Service
}

func NewCustomchannelsService(s *Service) *CustomchannelsService {
	rs := &CustomchannelsService{s: s}
	rs.Adunits = NewCustomchannelsAdunitsService(s)
	return rs
}

type CustomchannelsService struct {
	s *Service

	Adunits *CustomchannelsAdunitsService
}

func NewCustomchannelsAdunitsService(s *Service) *CustomchannelsAdunitsService {
	rs := &CustomchannelsAdunitsService{s: s}
	return rs
}

type CustomchannelsAdunitsService struct {
	s *Service
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	rs.Saved = NewReportsSavedService(s)
	return rs
}

type ReportsService struct {
	s *Service

	Saved *ReportsSavedService
}

func NewReportsSavedService(s *Service) *ReportsSavedService {
	rs := &ReportsSavedService{s: s}
	return rs
}

type ReportsSavedService struct {
	s *Service
}

func NewUrlchannelsService(s *Service) *UrlchannelsService {
	rs := &UrlchannelsService{s: s}
	return rs
}

type UrlchannelsService struct {
	s *Service
}

type AdClient struct {
	// ArcOptIn: Whether this ad client is opted in to ARC.
	ArcOptIn bool `json:"arcOptIn,omitempty"`

	// Id: Unique identifier of this ad client.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case
	// adexchangeseller#adClient.
	Kind string `json:"kind,omitempty"`

	// ProductCode: This ad client's product code, which corresponds to the
	// PRODUCT_CODE report dimension.
	ProductCode string `json:"productCode,omitempty"`

	// SupportsReporting: Whether this ad client supports being reported on.
	SupportsReporting bool `json:"supportsReporting,omitempty"`
}

type AdClients struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The ad clients returned in this list response.
	Items []*AdClient `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adexchangeseller#adClients.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through ad clients. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type AdUnit struct {
	// Code: Identity code of this ad unit, not necessarily unique across ad
	// clients.
	Code string `json:"code,omitempty"`

	// Id: Unique identifier of this ad unit. This should be considered an
	// opaque identifier; it is not safe to rely on it being in any
	// particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adexchangeseller#adUnit.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this ad unit.
	Name string `json:"name,omitempty"`

	// Status: Status of this ad unit. Possible values are:
	// NEW: Indicates
	// that the ad unit was created within the last seven days and does not
	// yet have any activity associated with it.
	//
	// ACTIVE: Indicates that
	// there has been activity on this ad unit in the last seven
	// days.
	//
	// INACTIVE: Indicates that there has been no activity on this ad
	// unit in the last seven days.
	Status string `json:"status,omitempty"`
}

type AdUnits struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The ad units returned in this list response.
	Items []*AdUnit `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adexchangeseller#adUnits.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through ad units. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type CustomChannel struct {
	// Code: Code of this custom channel, not necessarily unique across ad
	// clients.
	Code string `json:"code,omitempty"`

	// Id: Unique identifier of this custom channel. This should be
	// considered an opaque identifier; it is not safe to rely on it being
	// in any particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case
	// adexchangeseller#customChannel.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this custom channel.
	Name string `json:"name,omitempty"`

	// TargetingInfo: The targeting information of this custom channel, if
	// activated.
	TargetingInfo *CustomChannelTargetingInfo `json:"targetingInfo,omitempty"`
}

type CustomChannelTargetingInfo struct {
	// AdsAppearOn: The name used to describe this channel externally.
	AdsAppearOn string `json:"adsAppearOn,omitempty"`

	// Description: The external description of the channel.
	Description string `json:"description,omitempty"`

	// Location: The locations in which ads appear. (Only valid for content
	// and mobile content ads). Acceptable values for content ads are:
	// TOP_LEFT, TOP_CENTER, TOP_RIGHT, MIDDLE_LEFT, MIDDLE_CENTER,
	// MIDDLE_RIGHT, BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT,
	// MULTIPLE_LOCATIONS. Acceptable values for mobile content ads are:
	// TOP, MIDDLE, BOTTOM, MULTIPLE_LOCATIONS.
	Location string `json:"location,omitempty"`

	// SiteLanguage: The language of the sites ads will be displayed on.
	SiteLanguage string `json:"siteLanguage,omitempty"`
}

type CustomChannels struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The custom channels returned in this list response.
	Items []*CustomChannel `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case
	// adexchangeseller#customChannels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through custom
	// channels. To retrieve the next page of results, set the next
	// request's "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type Report struct {
	// Averages: The averages of the report. This is the same length as any
	// other row in the report; cells corresponding to dimension columns are
	// empty.
	Averages []string `json:"averages,omitempty"`

	// Headers: The header information of the columns requested in the
	// report. This is a list of headers; one for each dimension in the
	// request, followed by one for each metric in the request.
	Headers []*ReportHeaders `json:"headers,omitempty"`

	// Kind: Kind this is, in this case adexchangeseller#report.
	Kind string `json:"kind,omitempty"`

	// Rows: The output rows of the report. Each row is a list of cells; one
	// for each dimension in the request, followed by one for each metric in
	// the request. The dimension cells contain strings, and the metric
	// cells contain numbers.
	Rows [][]string `json:"rows,omitempty"`

	// TotalMatchedRows: The total number of rows matched by the report
	// request. Fewer rows may be returned in the response due to being
	// limited by the row count requested or the report row limit.
	TotalMatchedRows int64 `json:"totalMatchedRows,omitempty,string"`

	// Totals: The totals of the report. This is the same length as any
	// other row in the report; cells corresponding to dimension columns are
	// empty.
	Totals []string `json:"totals,omitempty"`

	// Warnings: Any warnings associated with generation of the report.
	Warnings []string `json:"warnings,omitempty"`
}

type ReportHeaders struct {
	// Currency: The currency of this column. Only present if the header
	// type is METRIC_CURRENCY.
	Currency string `json:"currency,omitempty"`

	// Name: The name of the header.
	Name string `json:"name,omitempty"`

	// Type: The type of the header; one of DIMENSION, METRIC_TALLY,
	// METRIC_RATIO, or METRIC_CURRENCY.
	Type string `json:"type,omitempty"`
}

type SavedReport struct {
	// Id: Unique identifier of this saved report.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case
	// adexchangeseller#savedReport.
	Kind string `json:"kind,omitempty"`

	// Name: This saved report's name.
	Name string `json:"name,omitempty"`
}

type SavedReports struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The saved reports returned in this list response.
	Items []*SavedReport `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case
	// adexchangeseller#savedReports.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through saved reports.
	// To retrieve the next page of results, set the next request's
	// "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type UrlChannel struct {
	// Id: Unique identifier of this URL channel. This should be considered
	// an opaque identifier; it is not safe to rely on it being in any
	// particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case
	// adexchangeseller#urlChannel.
	Kind string `json:"kind,omitempty"`

	// UrlPattern: URL Pattern of this URL channel. Does not include
	// "http://" or "https://". Example: www.example.com/home
	UrlPattern string `json:"urlPattern,omitempty"`
}

type UrlChannels struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The URL channels returned in this list response.
	Items []*UrlChannel `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case
	// adexchangeseller#urlChannels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through URL channels.
	// To retrieve the next page of results, set the next request's
	// "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

// method id "adexchangeseller.adclients.list":

type AdclientsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all ad clients in this Ad Exchange account.
func (r *AdclientsService) List() *AdclientsListCall {
	c := &AdclientsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad clients to include in the response, used for paging.
func (c *AdclientsListCall) MaxResults(maxResults int64) *AdclientsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad clients. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AdclientsListCall) PageToken(pageToken string) *AdclientsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AdclientsListCall) Do() (*AdClients, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdClients)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad clients in this Ad Exchange account.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.adclients.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of ad clients to include in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad clients. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients",
	//   "response": {
	//     "$ref": "AdClients"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.adunits.get":

type AdunitsGetCall struct {
	s          *Service
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// Get: Gets the specified ad unit in the specified ad client.
func (r *AdunitsService) Get(adClientId string, adUnitId string) *AdunitsGetCall {
	c := &AdunitsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

func (c *AdunitsGetCall) Do() (*AdUnit, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/adunits/{adUnitId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdUnit)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the specified ad unit in the specified ad client.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.adunits.get",
	//   "parameterOrder": [
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to get the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/adunits/{adUnitId}",
	//   "response": {
	//     "$ref": "AdUnit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.adunits.list":

type AdunitsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all ad units in the specified ad client for this Ad
// Exchange account.
func (r *AdunitsService) List(adClientId string) *AdunitsListCall {
	c := &AdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *AdunitsListCall) IncludeInactive(includeInactive bool) *AdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *AdunitsListCall) MaxResults(maxResults int64) *AdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AdunitsListCall) PageToken(pageToken string) *AdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AdunitsListCall) Do() (*AdUnits, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeInactive"]; ok {
		params.Set("includeInactive", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified ad client for this Ad Exchange account.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.adunits.list",
	//   "parameterOrder": [
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to list ad units.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeInactive": {
	//       "description": "Whether to include inactive ad units. Default: true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad units to include in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad units. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.adunits.customchannels.list":

type AdunitsCustomchannelsListCall struct {
	s          *Service
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// List: List all custom channels which the specified ad unit belongs
// to.
func (r *AdunitsCustomchannelsService) List(adClientId string, adUnitId string) *AdunitsCustomchannelsListCall {
	c := &AdunitsCustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *AdunitsCustomchannelsListCall) MaxResults(maxResults int64) *AdunitsCustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *AdunitsCustomchannelsListCall) PageToken(pageToken string) *AdunitsCustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AdunitsCustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/adunits/{adUnitId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels which the specified ad unit belongs to.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.adunits.customchannels.list",
	//   "parameterOrder": [
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client which contains the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit for which to list custom channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of custom channels to include in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through custom channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/adunits/{adUnitId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.customchannels.get":

type CustomchannelsGetCall struct {
	s               *Service
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// Get: Get the specified custom channel from the specified ad client.
func (r *CustomchannelsService) Get(adClientId string, customChannelId string) *CustomchannelsGetCall {
	c := &CustomchannelsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

func (c *CustomchannelsGetCall) Do() (*CustomChannel, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/customchannels/{customChannelId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CustomChannel)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get the specified custom channel from the specified ad client.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.customchannels.get",
	//   "parameterOrder": [
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client which contains the custom channel.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customChannelId": {
	//       "description": "Custom channel to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/customchannels/{customChannelId}",
	//   "response": {
	//     "$ref": "CustomChannel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.customchannels.list":

type CustomchannelsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all custom channels in the specified ad client for this Ad
// Exchange account.
func (r *CustomchannelsService) List(adClientId string) *CustomchannelsListCall {
	c := &CustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *CustomchannelsListCall) MaxResults(maxResults int64) *CustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *CustomchannelsListCall) PageToken(pageToken string) *CustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels in the specified ad client for this Ad Exchange account.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.customchannels.list",
	//   "parameterOrder": [
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to list custom channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of custom channels to include in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through custom channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.customchannels.adunits.list":

type CustomchannelsAdunitsListCall struct {
	s               *Service
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// List: List all ad units in the specified custom channel.
func (r *CustomchannelsAdunitsService) List(adClientId string, customChannelId string) *CustomchannelsAdunitsListCall {
	c := &CustomchannelsAdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *CustomchannelsAdunitsListCall) IncludeInactive(includeInactive bool) *CustomchannelsAdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *CustomchannelsAdunitsListCall) MaxResults(maxResults int64) *CustomchannelsAdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *CustomchannelsAdunitsListCall) PageToken(pageToken string) *CustomchannelsAdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CustomchannelsAdunitsListCall) Do() (*AdUnits, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeInactive"]; ok {
		params.Set("includeInactive", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/customchannels/{customChannelId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified custom channel.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.customchannels.adunits.list",
	//   "parameterOrder": [
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client which contains the custom channel.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customChannelId": {
	//       "description": "Custom channel for which to list ad units.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeInactive": {
	//       "description": "Whether to include inactive ad units. Default: true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad units to include in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad units. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/customchannels/{customChannelId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.reports.generate":

type ReportsGenerateCall struct {
	s         *Service
	startDate string
	endDate   string
	opt_      map[string]interface{}
}

// Generate: Generate an Ad Exchange report based on the report request
// sent in the query parameters. Returns the result as JSON; to retrieve
// output in CSV format specify "alt=csv" as a query parameter.
func (r *ReportsService) Generate(startDate string, endDate string) *ReportsGenerateCall {
	c := &ReportsGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.startDate = startDate
	c.endDate = endDate
	return c
}

// Dimension sets the optional parameter "dimension": Dimensions to base
// the report on.
func (c *ReportsGenerateCall) Dimension(dimension string) *ReportsGenerateCall {
	c.opt_["dimension"] = dimension
	return c
}

// Filter sets the optional parameter "filter": Filters to be run on the
// report.
func (c *ReportsGenerateCall) Filter(filter string) *ReportsGenerateCall {
	c.opt_["filter"] = filter
	return c
}

// Locale sets the optional parameter "locale": Optional locale to use
// for translating report output to a local language. Defaults to
// "en_US" if not specified.
func (c *ReportsGenerateCall) Locale(locale string) *ReportsGenerateCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of rows of report data to return.
func (c *ReportsGenerateCall) MaxResults(maxResults int64) *ReportsGenerateCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Metric sets the optional parameter "metric": Numeric columns to
// include in the report.
func (c *ReportsGenerateCall) Metric(metric string) *ReportsGenerateCall {
	c.opt_["metric"] = metric
	return c
}

// Sort sets the optional parameter "sort": The name of a dimension or
// metric to sort the resulting report on, optionally prefixed with "+"
// to sort ascending or "-" to sort descending. If no prefix is
// specified, the column is sorted ascending.
func (c *ReportsGenerateCall) Sort(sort string) *ReportsGenerateCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first row of report data to return.
func (c *ReportsGenerateCall) StartIndex(startIndex int64) *ReportsGenerateCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *ReportsGenerateCall) Do() (*Report, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("endDate", fmt.Sprintf("%v", c.endDate))
	params.Set("startDate", fmt.Sprintf("%v", c.startDate))
	if v, ok := c.opt_["dimension"]; ok {
		params.Set("dimension", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["metric"]; ok {
		params.Set("metric", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an Ad Exchange report based on the report request sent in the query parameters. Returns the result as JSON; to retrieve output in CSV format specify \"alt=csv\" as a query parameter.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.reports.generate",
	//   "parameterOrder": [
	//     "startDate",
	//     "endDate"
	//   ],
	//   "parameters": {
	//     "dimension": {
	//       "description": "Dimensions to base the report on.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "End of the date range to report on in \"YYYY-MM-DD\" format, inclusive.",
	//       "location": "query",
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,3}?)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filter": {
	//       "description": "Filters to be run on the report.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+(==|=@).+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "Optional locale to use for translating report output to a local language. Defaults to \"en_US\" if not specified.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of rows of report data to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "metric": {
	//       "description": "Numeric columns to include in the report.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "sort": {
	//       "description": "The name of a dimension or metric to sort the resulting report on, optionally prefixed with \"+\" to sort ascending or \"-\" to sort descending. If no prefix is specified, the column is sorted ascending.",
	//       "location": "query",
	//       "pattern": "(\\+|-)?[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "Start of the date range to report on in \"YYYY-MM-DD\" format, inclusive.",
	//       "location": "query",
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,3}?)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first row of report data to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "reports",
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "adexchangeseller.reports.saved.generate":

type ReportsSavedGenerateCall struct {
	s             *Service
	savedReportId string
	opt_          map[string]interface{}
}

// Generate: Generate an Ad Exchange report based on the saved report ID
// sent in the query parameters.
func (r *ReportsSavedService) Generate(savedReportId string) *ReportsSavedGenerateCall {
	c := &ReportsSavedGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.savedReportId = savedReportId
	return c
}

// Locale sets the optional parameter "locale": Optional locale to use
// for translating report output to a local language. Defaults to
// "en_US" if not specified.
func (c *ReportsSavedGenerateCall) Locale(locale string) *ReportsSavedGenerateCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of rows of report data to return.
func (c *ReportsSavedGenerateCall) MaxResults(maxResults int64) *ReportsSavedGenerateCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first row of report data to return.
func (c *ReportsSavedGenerateCall) StartIndex(startIndex int64) *ReportsSavedGenerateCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *ReportsSavedGenerateCall) Do() (*Report, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/{savedReportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{savedReportId}", url.QueryEscape(c.savedReportId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an Ad Exchange report based on the saved report ID sent in the query parameters.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.reports.saved.generate",
	//   "parameterOrder": [
	//     "savedReportId"
	//   ],
	//   "parameters": {
	//     "locale": {
	//       "description": "Optional locale to use for translating report output to a local language. Defaults to \"en_US\" if not specified.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of rows of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "50000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "savedReportId": {
	//       "description": "The saved report to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first row of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "reports/{savedReportId}",
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.reports.saved.list":

type ReportsSavedListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all saved reports in this Ad Exchange account.
func (r *ReportsSavedService) List() *ReportsSavedListCall {
	c := &ReportsSavedListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of saved reports to include in the response, used for paging.
func (c *ReportsSavedListCall) MaxResults(maxResults int64) *ReportsSavedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through saved reports. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *ReportsSavedListCall) PageToken(pageToken string) *ReportsSavedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *ReportsSavedListCall) Do() (*SavedReports, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/saved")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(SavedReports)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all saved reports in this Ad Exchange account.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.reports.saved.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of saved reports to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through saved reports. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "reports/saved",
	//   "response": {
	//     "$ref": "SavedReports"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}

// method id "adexchangeseller.urlchannels.list":

type UrlchannelsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all URL channels in the specified ad client for this Ad
// Exchange account.
func (r *UrlchannelsService) List(adClientId string) *UrlchannelsListCall {
	c := &UrlchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of URL channels to include in the response, used for paging.
func (c *UrlchannelsListCall) MaxResults(maxResults int64) *UrlchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through URL channels. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *UrlchannelsListCall) PageToken(pageToken string) *UrlchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *UrlchannelsListCall) Do() (*UrlChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/urlchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(UrlChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all URL channels in the specified ad client for this Ad Exchange account.",
	//   "httpMethod": "GET",
	//   "id": "adexchangeseller.urlchannels.list",
	//   "parameterOrder": [
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to list URL channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of URL channels to include in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through URL channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/urlchannels",
	//   "response": {
	//     "$ref": "UrlChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.seller",
	//     "https://www.googleapis.com/auth/adexchange.seller.readonly"
	//   ]
	// }

}
