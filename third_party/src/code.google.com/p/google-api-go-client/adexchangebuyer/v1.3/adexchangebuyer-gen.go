// Package adexchangebuyer provides access to the Ad Exchange Buyer API.
//
// See https://developers.google.com/ad-exchange/buyer-rest
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/adexchangebuyer/v1.3"
//   ...
//   adexchangebuyerService, err := adexchangebuyer.New(oauthHttpClient)
package adexchangebuyer

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

const apiId = "adexchangebuyer:v1.3"
const apiName = "adexchangebuyer"
const apiVersion = "v1.3"
const basePath = "https://www.googleapis.com/adexchangebuyer/v1.3/"

// OAuth2 scopes used by this API.
const (
	// Manage your Ad Exchange buyer account configuration
	AdexchangeBuyerScope = "https://www.googleapis.com/auth/adexchange.buyer"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Accounts = NewAccountsService(s)
	s.Creatives = NewCreativesService(s)
	s.DirectDeals = NewDirectDealsService(s)
	s.PerformanceReport = NewPerformanceReportService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Accounts *AccountsService

	Creatives *CreativesService

	DirectDeals *DirectDealsService

	PerformanceReport *PerformanceReportService
}

func NewAccountsService(s *Service) *AccountsService {
	rs := &AccountsService{s: s}
	return rs
}

type AccountsService struct {
	s *Service
}

func NewCreativesService(s *Service) *CreativesService {
	rs := &CreativesService{s: s}
	return rs
}

type CreativesService struct {
	s *Service
}

func NewDirectDealsService(s *Service) *DirectDealsService {
	rs := &DirectDealsService{s: s}
	return rs
}

type DirectDealsService struct {
	s *Service
}

func NewPerformanceReportService(s *Service) *PerformanceReportService {
	rs := &PerformanceReportService{s: s}
	return rs
}

type PerformanceReportService struct {
	s *Service
}

type Account struct {
	// BidderLocation: Your bidder locations that have distinct URLs.
	BidderLocation []*AccountBidderLocation `json:"bidderLocation,omitempty"`

	// CookieMatchingNid: The nid parameter value used in cookie match
	// requests. Please contact your technical account manager if you need
	// to change this.
	CookieMatchingNid string `json:"cookieMatchingNid,omitempty"`

	// CookieMatchingUrl: The base URL used in cookie match requests.
	CookieMatchingUrl string `json:"cookieMatchingUrl,omitempty"`

	// Id: Account id.
	Id int64 `json:"id,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// MaximumTotalQps: The sum of all bidderLocation.maximumQps values
	// cannot exceed this. Please contact your technical account manager if
	// you need to change this.
	MaximumTotalQps int64 `json:"maximumTotalQps,omitempty"`
}

type AccountBidderLocation struct {
	// MaximumQps: The maximum queries per second the Ad Exchange will send.
	MaximumQps int64 `json:"maximumQps,omitempty"`

	// Region: The geographical region the Ad Exchange should send requests
	// from. Only used by some quota systems, but always setting the value
	// is recommended. Allowed values:
	// - ASIA
	// - EUROPE
	// - US_EAST
	// -
	// US_WEST
	Region string `json:"region,omitempty"`

	// Url: The URL to which the Ad Exchange will send bid requests.
	Url string `json:"url,omitempty"`
}

type AccountsList struct {
	// Items: A list of accounts.
	Items []*Account `json:"items,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`
}

type Creative struct {
	// HTMLSnippet: The HTML snippet that displays the ad when inserted in
	// the web page. If set, videoURL should not be set.
	HTMLSnippet string `json:"HTMLSnippet,omitempty"`

	// AccountId: Account id.
	AccountId int64 `json:"accountId,omitempty"`

	// AdvertiserId: Detected advertiser id, if any. Read-only. This field
	// should not be set in requests.
	AdvertiserId googleapi.Int64s `json:"advertiserId,omitempty"`

	// AdvertiserName: The name of the company being advertised in the
	// creative.
	AdvertiserName string `json:"advertiserName,omitempty"`

	// AgencyId: The agency id for this creative.
	AgencyId int64 `json:"agencyId,omitempty,string"`

	// Attribute: All attributes for the ads that may be shown from this
	// snippet.
	Attribute []int64 `json:"attribute,omitempty"`

	// BuyerCreativeId: A buyer-specific id identifying the creative in this
	// ad.
	BuyerCreativeId string `json:"buyerCreativeId,omitempty"`

	// ClickThroughUrl: The set of destination urls for the snippet.
	ClickThroughUrl []string `json:"clickThroughUrl,omitempty"`

	// Corrections: Shows any corrections that were applied to this
	// creative. Read-only. This field should not be set in requests.
	Corrections []*CreativeCorrections `json:"corrections,omitempty"`

	// DisapprovalReasons: The reasons for disapproval, if any. Note that
	// not all disapproval reasons may be categorized, so it is possible for
	// the creative to have a status of DISAPPROVED with an empty list for
	// disapproval_reasons. In this case, please reach out to your TAM to
	// help debug the issue. Read-only. This field should not be set in
	// requests.
	DisapprovalReasons []*CreativeDisapprovalReasons `json:"disapprovalReasons,omitempty"`

	// FilteringReasons: The filtering reasons for the creative. Read-only.
	// This field should not be set in requests.
	FilteringReasons *CreativeFilteringReasons `json:"filteringReasons,omitempty"`

	// Height: Ad height.
	Height int64 `json:"height,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// ProductCategories: Detected product categories, if any. Read-only.
	// This field should not be set in requests.
	ProductCategories []int64 `json:"productCategories,omitempty"`

	// RestrictedCategories: All restricted categories for the ads that may
	// be shown from this snippet.
	RestrictedCategories []int64 `json:"restrictedCategories,omitempty"`

	// SensitiveCategories: Detected sensitive categories, if any.
	// Read-only. This field should not be set in requests.
	SensitiveCategories []int64 `json:"sensitiveCategories,omitempty"`

	// Status: Creative serving status. Read-only. This field should not be
	// set in requests.
	Status string `json:"status,omitempty"`

	// VendorType: All vendor types for the ads that may be shown from this
	// snippet.
	VendorType []int64 `json:"vendorType,omitempty"`

	// VideoURL: The url to fetch a video ad. If set, HTMLSnippet should not
	// be set.
	VideoURL string `json:"videoURL,omitempty"`

	// Width: Ad width.
	Width int64 `json:"width,omitempty"`
}

type CreativeCorrections struct {
	// Details: Additional details about the correction.
	Details []string `json:"details,omitempty"`

	// Reason: The type of correction that was applied to the creative.
	Reason string `json:"reason,omitempty"`
}

type CreativeDisapprovalReasons struct {
	// Details: Additional details about the reason for disapproval.
	Details []string `json:"details,omitempty"`

	// Reason: The categorized reason for disapproval.
	Reason string `json:"reason,omitempty"`
}

type CreativeFilteringReasons struct {
	// Date: The date in ISO 8601 format for the data. The data is collected
	// from 00:00:00 to 23:59:59 in PST.
	Date string `json:"date,omitempty"`

	// Reasons: The filtering reasons.
	Reasons []*CreativeFilteringReasonsReasons `json:"reasons,omitempty"`
}

type CreativeFilteringReasonsReasons struct {
	// FilteringCount: The number of times the creative was filtered for the
	// status. The count is aggregated across all publishers on the
	// exchange.
	FilteringCount int64 `json:"filteringCount,omitempty,string"`

	// FilteringStatus: The filtering status code. Please refer to
	// "creative-status-codes.txt" in the Downloads section for the status
	// codes.
	FilteringStatus int64 `json:"filteringStatus,omitempty"`
}

type CreativesList struct {
	// Items: A list of creatives.
	Items []*Creative `json:"items,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through creatives. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type DirectDeal struct {
	// AccountId: The account id of the buyer this deal is for.
	AccountId int64 `json:"accountId,omitempty"`

	// Advertiser: The name of the advertiser this deal is for.
	Advertiser string `json:"advertiser,omitempty"`

	// CurrencyCode: The currency code that applies to the fixed_cpm value.
	// If not set then assumed to be USD.
	CurrencyCode string `json:"currencyCode,omitempty"`

	// EndTime: End time for when this deal stops being active. If not set
	// then this deal is valid until manually disabled by the publisher. In
	// seconds since the epoch.
	EndTime int64 `json:"endTime,omitempty,string"`

	// FixedCpm: The fixed price for this direct deal. In cpm micros of
	// currency according to currency_code. If set, then this deal is
	// eligible for the fixed price tier of buying (highest priority, pay
	// exactly the configured fixed price).
	FixedCpm int64 `json:"fixedCpm,omitempty,string"`

	// Id: Deal id.
	Id int64 `json:"id,omitempty,string"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// Name: Deal name.
	Name string `json:"name,omitempty"`

	// PrivateExchangeMinCpm: The minimum price for this direct deal. In cpm
	// micros of currency according to currency_code. If set, then this deal
	// is eligible for the private exchange tier of buying (below fixed
	// price priority, run as a second price auction).
	PrivateExchangeMinCpm int64 `json:"privateExchangeMinCpm,omitempty,string"`

	// SellerNetwork: The name of the publisher offering this direct deal.
	SellerNetwork string `json:"sellerNetwork,omitempty"`

	// StartTime: Start time for when this deal becomes active. If not set
	// then this deal is active immediately upon creation. In seconds since
	// the epoch.
	StartTime int64 `json:"startTime,omitempty,string"`
}

type DirectDealsList struct {
	// DirectDeals: A list of direct deals relevant for your account.
	DirectDeals []*DirectDeal `json:"directDeals,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`
}

type PerformanceReport struct {
	// CalloutStatusRate: Rate of various prefiltering statuses per match.
	// Please refer to the callout-status-codes.txt file for different
	// statuses.
	CalloutStatusRate []interface{} `json:"calloutStatusRate,omitempty"`

	// CookieMatcherStatusRate: Average QPS for cookie matcher operations.
	CookieMatcherStatusRate []interface{} `json:"cookieMatcherStatusRate,omitempty"`

	// CreativeStatusRate: Rate of ads with a given status. Please refer to
	// the creative-status-codes.txt file for different statuses.
	CreativeStatusRate []interface{} `json:"creativeStatusRate,omitempty"`

	// HostedMatchStatusRate: Average QPS for hosted match operations.
	HostedMatchStatusRate []interface{} `json:"hostedMatchStatusRate,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// Latency50thPercentile: The 50th percentile round trip latency(ms) as
	// perceived from Google servers for the duration period covered by the
	// report.
	Latency50thPercentile float64 `json:"latency50thPercentile,omitempty"`

	// Latency85thPercentile: The 85th percentile round trip latency(ms) as
	// perceived from Google servers for the duration period covered by the
	// report.
	Latency85thPercentile float64 `json:"latency85thPercentile,omitempty"`

	// Latency95thPercentile: The 95th percentile round trip latency(ms) as
	// perceived from Google servers for the duration period covered by the
	// report.
	Latency95thPercentile float64 `json:"latency95thPercentile,omitempty"`

	// NoQuotaInRegion: Rate of various quota account statuses per quota
	// check.
	NoQuotaInRegion float64 `json:"noQuotaInRegion,omitempty"`

	// OutOfQuota: Rate of various quota account statuses per quota check.
	OutOfQuota float64 `json:"outOfQuota,omitempty"`

	// PixelMatchRequests: Average QPS for pixel match requests from
	// clients.
	PixelMatchRequests float64 `json:"pixelMatchRequests,omitempty"`

	// PixelMatchResponses: Average QPS for pixel match responses from
	// clients.
	PixelMatchResponses float64 `json:"pixelMatchResponses,omitempty"`

	// QuotaConfiguredLimit: The configured quota limits for this account.
	QuotaConfiguredLimit float64 `json:"quotaConfiguredLimit,omitempty"`

	// QuotaThrottledLimit: The throttled quota limits for this account.
	QuotaThrottledLimit float64 `json:"quotaThrottledLimit,omitempty"`

	// Region: The trading location of this data.
	Region string `json:"region,omitempty"`

	// Timestamp: The unix timestamp of the starting time of this
	// performance data.
	Timestamp int64 `json:"timestamp,omitempty,string"`
}

type PerformanceReportList struct {
	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// PerformanceReport: A list of performance reports relevant for the
	// account.
	PerformanceReport []*PerformanceReport `json:"performanceReport,omitempty"`
}

// method id "adexchangebuyer.accounts.get":

type AccountsGetCall struct {
	s    *Service
	id   int64
	opt_ map[string]interface{}
}

// Get: Gets one account by ID.
func (r *AccountsService) Get(id int64) *AccountsGetCall {
	c := &AccountsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *AccountsGetCall) Do() (*Account, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", strconv.FormatInt(c.id, 10), 1)
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
	ret := new(Account)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one account by ID.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.accounts.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The account id",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "accounts/{id}",
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.accounts.list":

type AccountsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves the authenticated user's list of accounts.
func (r *AccountsService) List() *AccountsListCall {
	c := &AccountsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *AccountsListCall) Do() (*AccountsList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts")
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
	ret := new(AccountsList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the authenticated user's list of accounts.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.accounts.list",
	//   "path": "accounts",
	//   "response": {
	//     "$ref": "AccountsList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.accounts.patch":

type AccountsPatchCall struct {
	s       *Service
	id      int64
	account *Account
	opt_    map[string]interface{}
}

// Patch: Updates an existing account. This method supports patch
// semantics.
func (r *AccountsService) Patch(id int64, account *Account) *AccountsPatchCall {
	c := &AccountsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.account = account
	return c
}

func (c *AccountsPatchCall) Do() (*Account, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", strconv.FormatInt(c.id, 10), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Account)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing account. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "adexchangebuyer.accounts.patch",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The account id",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "accounts/{id}",
	//   "request": {
	//     "$ref": "Account"
	//   },
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.accounts.update":

type AccountsUpdateCall struct {
	s       *Service
	id      int64
	account *Account
	opt_    map[string]interface{}
}

// Update: Updates an existing account.
func (r *AccountsService) Update(id int64, account *Account) *AccountsUpdateCall {
	c := &AccountsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.account = account
	return c
}

func (c *AccountsUpdateCall) Do() (*Account, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", strconv.FormatInt(c.id, 10), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Account)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing account.",
	//   "httpMethod": "PUT",
	//   "id": "adexchangebuyer.accounts.update",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The account id",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "accounts/{id}",
	//   "request": {
	//     "$ref": "Account"
	//   },
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.creatives.get":

type CreativesGetCall struct {
	s               *Service
	accountId       int64
	buyerCreativeId string
	opt_            map[string]interface{}
}

// Get: Gets the status for a single creative. A creative will be
// available 30-40 minutes after submission.
func (r *CreativesService) Get(accountId int64, buyerCreativeId string) *CreativesGetCall {
	c := &CreativesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.buyerCreativeId = buyerCreativeId
	return c
}

func (c *CreativesGetCall) Do() (*Creative, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives/{accountId}/{buyerCreativeId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", strconv.FormatInt(c.accountId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{buyerCreativeId}", url.QueryEscape(c.buyerCreativeId), 1)
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
	ret := new(Creative)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the status for a single creative. A creative will be available 30-40 minutes after submission.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.creatives.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "buyerCreativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The id for the account that will serve this creative.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "buyerCreativeId": {
	//       "description": "The buyer-specific id for this creative.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "creatives/{accountId}/{buyerCreativeId}",
	//   "response": {
	//     "$ref": "Creative"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.creatives.insert":

type CreativesInsertCall struct {
	s        *Service
	creative *Creative
	opt_     map[string]interface{}
}

// Insert: Submit a new creative.
func (r *CreativesService) Insert(creative *Creative) *CreativesInsertCall {
	c := &CreativesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.creative = creative
	return c
}

func (c *CreativesInsertCall) Do() (*Creative, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.creative)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Creative)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Submit a new creative.",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.creatives.insert",
	//   "path": "creatives",
	//   "request": {
	//     "$ref": "Creative"
	//   },
	//   "response": {
	//     "$ref": "Creative"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.creatives.list":

type CreativesListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of the authenticated user's active creatives.
// A creative will be available 30-40 minutes after submission.
func (r *CreativesService) List() *CreativesListCall {
	c := &CreativesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of entries returned on one result page. If not set, the default is
// 100.
func (c *CreativesListCall) MaxResults(maxResults int64) *CreativesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad clients. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *CreativesListCall) PageToken(pageToken string) *CreativesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// StatusFilter sets the optional parameter "statusFilter": When
// specified, only creatives having the given status are returned.
func (c *CreativesListCall) StatusFilter(statusFilter string) *CreativesListCall {
	c.opt_["statusFilter"] = statusFilter
	return c
}

func (c *CreativesListCall) Do() (*CreativesList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["statusFilter"]; ok {
		params.Set("statusFilter", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives")
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
	ret := new(CreativesList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of the authenticated user's active creatives. A creative will be available 30-40 minutes after submission.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.creatives.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of entries returned on one result page. If not set, the default is 100. Optional.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad clients. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "statusFilter": {
	//       "description": "When specified, only creatives having the given status are returned.",
	//       "enum": [
	//         "approved",
	//         "disapproved",
	//         "not_checked"
	//       ],
	//       "enumDescriptions": [
	//         "Creatives which have been approved.",
	//         "Creatives which have been disapproved.",
	//         "Creatives whose status is not yet checked."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "creatives",
	//   "response": {
	//     "$ref": "CreativesList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.directDeals.get":

type DirectDealsGetCall struct {
	s    *Service
	id   int64
	opt_ map[string]interface{}
}

// Get: Gets one direct deal by ID.
func (r *DirectDealsService) Get(id int64) *DirectDealsGetCall {
	c := &DirectDealsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *DirectDealsGetCall) Do() (*DirectDeal, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "directdeals/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", strconv.FormatInt(c.id, 10), 1)
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
	ret := new(DirectDeal)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one direct deal by ID.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.directDeals.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The direct deal id",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "directdeals/{id}",
	//   "response": {
	//     "$ref": "DirectDeal"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.directDeals.list":

type DirectDealsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves the authenticated user's list of direct deals.
func (r *DirectDealsService) List() *DirectDealsListCall {
	c := &DirectDealsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *DirectDealsListCall) Do() (*DirectDealsList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "directdeals")
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
	ret := new(DirectDealsList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the authenticated user's list of direct deals.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.directDeals.list",
	//   "path": "directdeals",
	//   "response": {
	//     "$ref": "DirectDealsList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.performanceReport.list":

type PerformanceReportListCall struct {
	s             *Service
	accountId     int64
	endDateTime   string
	startDateTime string
	opt_          map[string]interface{}
}

// List: Retrieves the authenticated user's list of performance metrics.
func (r *PerformanceReportService) List(accountId int64, endDateTime string, startDateTime string) *PerformanceReportListCall {
	c := &PerformanceReportListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.endDateTime = endDateTime
	c.startDateTime = startDateTime
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of entries returned on one result page. If not set, the default is
// 100.
func (c *PerformanceReportListCall) MaxResults(maxResults int64) *PerformanceReportListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through performance reports. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *PerformanceReportListCall) PageToken(pageToken string) *PerformanceReportListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *PerformanceReportListCall) Do() (*PerformanceReportList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("accountId", fmt.Sprintf("%v", c.accountId))
	params.Set("endDateTime", fmt.Sprintf("%v", c.endDateTime))
	params.Set("startDateTime", fmt.Sprintf("%v", c.startDateTime))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "performancereport")
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
	ret := new(PerformanceReportList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the authenticated user's list of performance metrics.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.performanceReport.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "endDateTime",
	//     "startDateTime"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to get the reports.",
	//       "format": "int64",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "endDateTime": {
	//       "description": "The end time of the report in ISO 8601 timestamp format using UTC.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of entries returned on one result page. If not set, the default is 100. Optional.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through performance reports. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startDateTime": {
	//       "description": "The start time of the report in ISO 8601 timestamp format using UTC.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "performancereport",
	//   "response": {
	//     "$ref": "PerformanceReportList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}
