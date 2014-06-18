// Package adsense provides access to the AdSense Management API.
//
// See https://developers.google.com/adsense/management/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/adsense/v1.1"
//   ...
//   adsenseService, err := adsense.New(oauthHttpClient)
package adsense

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

const apiId = "adsense:v1.1"
const apiName = "adsense"
const apiVersion = "v1.1"
const basePath = "https://www.googleapis.com/adsense/v1.1/"

// OAuth2 scopes used by this API.
const (
	// View and manage your AdSense data
	AdsenseScope = "https://www.googleapis.com/auth/adsense"

	// View your AdSense data
	AdsenseReadonlyScope = "https://www.googleapis.com/auth/adsense.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client}
	s.Accounts = NewAccountsService(s)
	s.Adclients = NewAdclientsService(s)
	s.Adunits = NewAdunitsService(s)
	s.Customchannels = NewCustomchannelsService(s)
	s.Reports = NewReportsService(s)
	s.Urlchannels = NewUrlchannelsService(s)
	return s, nil
}

type Service struct {
	client *http.Client

	Accounts *AccountsService

	Adclients *AdclientsService

	Adunits *AdunitsService

	Customchannels *CustomchannelsService

	Reports *ReportsService

	Urlchannels *UrlchannelsService
}

func NewAccountsService(s *Service) *AccountsService {
	rs := &AccountsService{s: s}
	rs.Adclients = NewAccountsAdclientsService(s)
	rs.Adunits = NewAccountsAdunitsService(s)
	rs.Customchannels = NewAccountsCustomchannelsService(s)
	rs.Reports = NewAccountsReportsService(s)
	rs.Urlchannels = NewAccountsUrlchannelsService(s)
	return rs
}

type AccountsService struct {
	s *Service

	Adclients *AccountsAdclientsService

	Adunits *AccountsAdunitsService

	Customchannels *AccountsCustomchannelsService

	Reports *AccountsReportsService

	Urlchannels *AccountsUrlchannelsService
}

func NewAccountsAdclientsService(s *Service) *AccountsAdclientsService {
	rs := &AccountsAdclientsService{s: s}
	return rs
}

type AccountsAdclientsService struct {
	s *Service
}

func NewAccountsAdunitsService(s *Service) *AccountsAdunitsService {
	rs := &AccountsAdunitsService{s: s}
	rs.Customchannels = NewAccountsAdunitsCustomchannelsService(s)
	return rs
}

type AccountsAdunitsService struct {
	s *Service

	Customchannels *AccountsAdunitsCustomchannelsService
}

func NewAccountsAdunitsCustomchannelsService(s *Service) *AccountsAdunitsCustomchannelsService {
	rs := &AccountsAdunitsCustomchannelsService{s: s}
	return rs
}

type AccountsAdunitsCustomchannelsService struct {
	s *Service
}

func NewAccountsCustomchannelsService(s *Service) *AccountsCustomchannelsService {
	rs := &AccountsCustomchannelsService{s: s}
	rs.Adunits = NewAccountsCustomchannelsAdunitsService(s)
	return rs
}

type AccountsCustomchannelsService struct {
	s *Service

	Adunits *AccountsCustomchannelsAdunitsService
}

func NewAccountsCustomchannelsAdunitsService(s *Service) *AccountsCustomchannelsAdunitsService {
	rs := &AccountsCustomchannelsAdunitsService{s: s}
	return rs
}

type AccountsCustomchannelsAdunitsService struct {
	s *Service
}

func NewAccountsReportsService(s *Service) *AccountsReportsService {
	rs := &AccountsReportsService{s: s}
	return rs
}

type AccountsReportsService struct {
	s *Service
}

func NewAccountsUrlchannelsService(s *Service) *AccountsUrlchannelsService {
	rs := &AccountsUrlchannelsService{s: s}
	return rs
}

type AccountsUrlchannelsService struct {
	s *Service
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
	return rs
}

type ReportsService struct {
	s *Service
}

func NewUrlchannelsService(s *Service) *UrlchannelsService {
	rs := &UrlchannelsService{s: s}
	return rs
}

type UrlchannelsService struct {
	s *Service
}

type Account struct {
	// Id: Unique identifier of this account.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#account.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this account.
	Name string `json:"name,omitempty"`

	// SubAccounts: Sub accounts of the this account.
	SubAccounts []*Account `json:"subAccounts,omitempty"`
}

type Accounts struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The accounts returned in this list response.
	Items []*Account `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#accounts.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through accounts. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type AdClient struct {
	// Id: Unique identifier of this ad client.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#adClient.
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

	// Kind: Kind of list this is, in this case adsense#adClients.
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

	// Kind: Kind of resource this is, in this case adsense#adUnit.
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

	// Kind: Kind of list this is, in this case adsense#adUnits.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through ad units. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type AdsenseReportsGenerateResponse struct {
	// Averages: The averages of the report. This is the same length as any
	// other row in the report; cells corresponding to dimension columns are
	// empty.
	Averages []string `json:"averages,omitempty"`

	// Headers: The header information of the columns requested in the
	// report. This is a list of headers; one for each dimension in the
	// request, followed by one for each metric in the request.
	Headers []*AdsenseReportsGenerateResponseHeaders `json:"headers,omitempty"`

	// Kind: Kind this is, in this case adsense#report.
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

type AdsenseReportsGenerateResponseHeaders struct {
	// Currency: The currency of this column. Only present if the header
	// type is METRIC_CURRENCY.
	Currency string `json:"currency,omitempty"`

	// Name: The name of the header.
	Name string `json:"name,omitempty"`

	// Type: The type of the header; one of DIMENSION, METRIC_TALLY,
	// METRIC_RATIO, or METRIC_CURRENCY.
	Type string `json:"type,omitempty"`
}

type CustomChannel struct {
	// Code: Code of this custom channel, not necessarily unique across ad
	// clients.
	Code string `json:"code,omitempty"`

	// Id: Unique identifier of this custom channel. This should be
	// considered an opaque identifier; it is not safe to rely on it being
	// in any particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#customChannel.
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

	// Kind: Kind of list this is, in this case adsense#customChannels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through custom
	// channels. To retrieve the next page of results, set the next
	// request's "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type UrlChannel struct {
	// Id: Unique identifier of this URL channel. This should be considered
	// an opaque identifier; it is not safe to rely on it being in any
	// particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#urlChannel.
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

	// Kind: Kind of list this is, in this case adsense#urlChannels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through URL channels.
	// To retrieve the next page of results, set the next request's
	// "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

// method id "adsense.accounts.get":

type AccountsGetCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// Get: Get information about the selected AdSense account.
func (r *AccountsService) Get(accountId string) *AccountsGetCall {
	c := &AccountsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// Tree sets the optional parameter "tree": Whether the tree of sub
// accounts should be returned.
func (c *AccountsGetCall) Tree(tree bool) *AccountsGetCall {
	c.opt_["tree"] = tree
	return c
}

func (c *AccountsGetCall) Do() (*Account, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["tree"]; ok {
		params.Set("tree", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Account)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get information about the selected AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.get",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to get information about.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tree": {
	//       "description": "Whether the tree of sub accounts should be returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "accounts/{accountId}",
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.list":

type AccountsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all accounts available to this AdSense account.
func (r *AccountsService) List() *AccountsListCall {
	c := &AccountsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of accounts to include in the response, used for paging.
func (c *AccountsListCall) MaxResults(maxResults int64) *AccountsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through accounts. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsListCall) PageToken(pageToken string) *AccountsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsListCall) Do() (*Accounts, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Accounts)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all accounts available to this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of accounts to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through accounts. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts",
	//   "response": {
	//     "$ref": "Accounts"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adclients.list":

type AccountsAdclientsListCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// List: List all ad clients in the specified account.
func (r *AccountsAdclientsService) List(accountId string) *AccountsAdclientsListCall {
	c := &AccountsAdclientsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad clients to include in the response, used for paging.
func (c *AccountsAdclientsListCall) MaxResults(maxResults int64) *AccountsAdclientsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad clients. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsAdclientsListCall) PageToken(pageToken string) *AccountsAdclientsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsAdclientsListCall) Do() (*AdClients, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdClients)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad clients in the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adclients.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account for which to list ad clients.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad clients to include in the response, used for paging.",
	//       "format": "int32",
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
	//   "path": "accounts/{accountId}/adclients",
	//   "response": {
	//     "$ref": "AdClients"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adunits.get":

type AccountsAdunitsGetCall struct {
	s          *Service
	accountId  string
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// Get: Gets the specified ad unit in the specified ad client for the
// specified account.
func (r *AccountsAdunitsService) Get(accountId string, adClientId string, adUnitId string) *AccountsAdunitsGetCall {
	c := &AccountsAdunitsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

func (c *AccountsAdunitsGetCall) Do() (*AdUnit, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdUnit)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the specified ad unit in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adunits.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
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
	//   "path": "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}",
	//   "response": {
	//     "$ref": "AdUnit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adunits.list":

type AccountsAdunitsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	opt_       map[string]interface{}
}

// List: List all ad units in the specified ad client for the specified
// account.
func (r *AccountsAdunitsService) List(accountId string, adClientId string) *AccountsAdunitsListCall {
	c := &AccountsAdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *AccountsAdunitsListCall) IncludeInactive(includeInactive bool) *AccountsAdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *AccountsAdunitsListCall) MaxResults(maxResults int64) *AccountsAdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsAdunitsListCall) PageToken(pageToken string) *AccountsAdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsAdunitsListCall) Do() (*AdUnits, error) {
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients/{adClientId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adunits.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
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
	//       "format": "int32",
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
	//   "path": "accounts/{accountId}/adclients/{adClientId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adunits.customchannels.list":

type AccountsAdunitsCustomchannelsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// List: List all custom channels which the specified ad unit belongs
// to.
func (r *AccountsAdunitsCustomchannelsService) List(accountId string, adClientId string, adUnitId string) *AccountsAdunitsCustomchannelsListCall {
	c := &AccountsAdunitsCustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *AccountsAdunitsCustomchannelsListCall) MaxResults(maxResults int64) *AccountsAdunitsCustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *AccountsAdunitsCustomchannelsListCall) PageToken(pageToken string) *AccountsAdunitsCustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsAdunitsCustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
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
	//   "id": "adsense.accounts.adunits.customchannels.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
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
	//       "format": "int32",
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
	//   "path": "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.customchannels.get":

type AccountsCustomchannelsGetCall struct {
	s               *Service
	accountId       string
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// Get: Get the specified custom channel from the specified ad client
// for the specified account.
func (r *AccountsCustomchannelsService) Get(accountId string, adClientId string, customChannelId string) *AccountsCustomchannelsGetCall {
	c := &AccountsCustomchannelsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

func (c *AccountsCustomchannelsGetCall) Do() (*CustomChannel, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CustomChannel)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get the specified custom channel from the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.customchannels.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
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
	//   "path": "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}",
	//   "response": {
	//     "$ref": "CustomChannel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.customchannels.list":

type AccountsCustomchannelsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	opt_       map[string]interface{}
}

// List: List all custom channels in the specified ad client for the
// specified account.
func (r *AccountsCustomchannelsService) List(accountId string, adClientId string) *AccountsCustomchannelsListCall {
	c := &AccountsCustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *AccountsCustomchannelsListCall) MaxResults(maxResults int64) *AccountsCustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *AccountsCustomchannelsListCall) PageToken(pageToken string) *AccountsCustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsCustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients/{adClientId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.customchannels.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client for which to list custom channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of custom channels to include in the response, used for paging.",
	//       "format": "int32",
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
	//   "path": "accounts/{accountId}/adclients/{adClientId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.customchannels.adunits.list":

type AccountsCustomchannelsAdunitsListCall struct {
	s               *Service
	accountId       string
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// List: List all ad units in the specified custom channel.
func (r *AccountsCustomchannelsAdunitsService) List(accountId string, adClientId string, customChannelId string) *AccountsCustomchannelsAdunitsListCall {
	c := &AccountsCustomchannelsAdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *AccountsCustomchannelsAdunitsListCall) IncludeInactive(includeInactive bool) *AccountsCustomchannelsAdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *AccountsCustomchannelsAdunitsListCall) MaxResults(maxResults int64) *AccountsCustomchannelsAdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsCustomchannelsAdunitsListCall) PageToken(pageToken string) *AccountsCustomchannelsAdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsCustomchannelsAdunitsListCall) Do() (*AdUnits, error) {
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
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
	//   "id": "adsense.accounts.customchannels.adunits.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
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
	//       "format": "int32",
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
	//   "path": "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.reports.generate":

type AccountsReportsGenerateCall struct {
	s         *Service
	accountId string
	startDate string
	endDate   string
	opt_      map[string]interface{}
}

// Generate: Generate an AdSense report based on the report request sent
// in the query parameters. Returns the result as JSON; to retrieve
// output in CSV format specify "alt=csv" as a query parameter.
func (r *AccountsReportsService) Generate(accountId string, startDate string, endDate string) *AccountsReportsGenerateCall {
	c := &AccountsReportsGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.startDate = startDate
	c.endDate = endDate
	return c
}

// Currency sets the optional parameter "currency": Optional currency to
// use when reporting on monetary metrics. Defaults to the account's
// currency if not set.
func (c *AccountsReportsGenerateCall) Currency(currency string) *AccountsReportsGenerateCall {
	c.opt_["currency"] = currency
	return c
}

// Dimension sets the optional parameter "dimension": Dimensions to base
// the report on.
func (c *AccountsReportsGenerateCall) Dimension(dimension string) *AccountsReportsGenerateCall {
	c.opt_["dimension"] = dimension
	return c
}

// Filter sets the optional parameter "filter": Filters to be run on the
// report.
func (c *AccountsReportsGenerateCall) Filter(filter string) *AccountsReportsGenerateCall {
	c.opt_["filter"] = filter
	return c
}

// Locale sets the optional parameter "locale": Optional locale to use
// for translating report output to a local language. Defaults to
// "en_US" if not specified.
func (c *AccountsReportsGenerateCall) Locale(locale string) *AccountsReportsGenerateCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of rows of report data to return.
func (c *AccountsReportsGenerateCall) MaxResults(maxResults int64) *AccountsReportsGenerateCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Metric sets the optional parameter "metric": Numeric columns to
// include in the report.
func (c *AccountsReportsGenerateCall) Metric(metric string) *AccountsReportsGenerateCall {
	c.opt_["metric"] = metric
	return c
}

// Sort sets the optional parameter "sort": The name of a dimension or
// metric to sort the resulting report on, optionally prefixed with "+"
// to sort ascending or "-" to sort descending. If no prefix is
// specified, the column is sorted ascending.
func (c *AccountsReportsGenerateCall) Sort(sort string) *AccountsReportsGenerateCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first row of report data to return.
func (c *AccountsReportsGenerateCall) StartIndex(startIndex int64) *AccountsReportsGenerateCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *AccountsReportsGenerateCall) Do() (*AdsenseReportsGenerateResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("endDate", fmt.Sprintf("%v", c.endDate))
	params.Set("startDate", fmt.Sprintf("%v", c.startDate))
	if v, ok := c.opt_["currency"]; ok {
		params.Set("currency", fmt.Sprintf("%v", v))
	}
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdsenseReportsGenerateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an AdSense report based on the report request sent in the query parameters. Returns the result as JSON; to retrieve output in CSV format specify \"alt=csv\" as a query parameter.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.reports.generate",
	//   "parameterOrder": [
	//     "accountId",
	//     "startDate",
	//     "endDate"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account upon which to report.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "currency": {
	//       "description": "Optional currency to use when reporting on monetary metrics. Defaults to the account's currency if not set.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z]+",
	//       "type": "string"
	//     },
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
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,2}?)",
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
	//       "format": "int32",
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
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,2}?)",
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
	//   "path": "accounts/{accountId}/reports",
	//   "response": {
	//     "$ref": "AdsenseReportsGenerateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "adsense.accounts.urlchannels.list":

type AccountsUrlchannelsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	opt_       map[string]interface{}
}

// List: List all URL channels in the specified ad client for the
// specified account.
func (r *AccountsUrlchannelsService) List(accountId string, adClientId string) *AccountsUrlchannelsListCall {
	c := &AccountsUrlchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of URL channels to include in the response, used for paging.
func (c *AccountsUrlchannelsListCall) MaxResults(maxResults int64) *AccountsUrlchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through URL channels. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsUrlchannelsListCall) PageToken(pageToken string) *AccountsUrlchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsUrlchannelsListCall) Do() (*UrlChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "accounts/{accountId}/adclients/{adClientId}/urlchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(UrlChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all URL channels in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.urlchannels.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client for which to list URL channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of URL channels to include in the response, used for paging.",
	//       "format": "int32",
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
	//   "path": "accounts/{accountId}/adclients/{adClientId}/urlchannels",
	//   "response": {
	//     "$ref": "UrlChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adclients.list":

type AdclientsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all ad clients in this AdSense account.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdClients)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad clients in this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.adclients.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of ad clients to include in the response, used for paging.",
	//       "format": "int32",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adunits.get":

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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients/{adClientId}/adunits/{adUnitId}")
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
	defer res.Body.Close()
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
	//   "id": "adsense.adunits.get",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adunits.list":

type AdunitsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all ad units in the specified ad client for this AdSense
// account.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients/{adClientId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified ad client for this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.adunits.list",
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
	//       "format": "int32",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adunits.customchannels.list":

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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients/{adClientId}/adunits/{adUnitId}/customchannels")
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
	defer res.Body.Close()
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
	//   "id": "adsense.adunits.customchannels.list",
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
	//       "format": "int32",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.customchannels.get":

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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients/{adClientId}/customchannels/{customChannelId}")
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
	defer res.Body.Close()
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
	//   "id": "adsense.customchannels.get",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.customchannels.list":

type CustomchannelsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all custom channels in the specified ad client for this
// AdSense account.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients/{adClientId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels in the specified ad client for this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.customchannels.list",
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
	//       "format": "int32",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.customchannels.adunits.list":

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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients/{adClientId}/customchannels/{customChannelId}/adunits")
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
	defer res.Body.Close()
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
	//   "id": "adsense.customchannels.adunits.list",
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
	//       "format": "int32",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.reports.generate":

type ReportsGenerateCall struct {
	s         *Service
	startDate string
	endDate   string
	opt_      map[string]interface{}
}

// Generate: Generate an AdSense report based on the report request sent
// in the query parameters. Returns the result as JSON; to retrieve
// output in CSV format specify "alt=csv" as a query parameter.
func (r *ReportsService) Generate(startDate string, endDate string) *ReportsGenerateCall {
	c := &ReportsGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.startDate = startDate
	c.endDate = endDate
	return c
}

// AccountId sets the optional parameter "accountId": Accounts upon
// which to report.
func (c *ReportsGenerateCall) AccountId(accountId string) *ReportsGenerateCall {
	c.opt_["accountId"] = accountId
	return c
}

// Currency sets the optional parameter "currency": Optional currency to
// use when reporting on monetary metrics. Defaults to the account's
// currency if not set.
func (c *ReportsGenerateCall) Currency(currency string) *ReportsGenerateCall {
	c.opt_["currency"] = currency
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

func (c *ReportsGenerateCall) Do() (*AdsenseReportsGenerateResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("endDate", fmt.Sprintf("%v", c.endDate))
	params.Set("startDate", fmt.Sprintf("%v", c.startDate))
	if v, ok := c.opt_["accountId"]; ok {
		params.Set("accountId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["currency"]; ok {
		params.Set("currency", fmt.Sprintf("%v", v))
	}
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AdsenseReportsGenerateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an AdSense report based on the report request sent in the query parameters. Returns the result as JSON; to retrieve output in CSV format specify \"alt=csv\" as a query parameter.",
	//   "httpMethod": "GET",
	//   "id": "adsense.reports.generate",
	//   "parameterOrder": [
	//     "startDate",
	//     "endDate"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Accounts upon which to report.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "currency": {
	//       "description": "Optional currency to use when reporting on monetary metrics. Defaults to the account's currency if not set.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z]+",
	//       "type": "string"
	//     },
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
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,2}?)",
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
	//       "format": "int32",
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
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,2}?)",
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
	//   "path": "reports",
	//   "response": {
	//     "$ref": "AdsenseReportsGenerateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "adsense.urlchannels.list":

type UrlchannelsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all URL channels in the specified ad client for this
// AdSense account.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/adsense/v1.1/", "adclients/{adClientId}/urlchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(UrlChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all URL channels in the specified ad client for this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.urlchannels.list",
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
	//       "format": "int32",
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
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}
