// Package adexchangebuyer provides access to the Ad Exchange Buyer API.
//
// See https://developers.google.com/ad-exchange/buyer-rest
//
// Usage example:
//
//   import "google.golang.org/api/adexchangebuyer/v1.4"
//   ...
//   adexchangebuyerService, err := adexchangebuyer.New(oauthHttpClient)
package adexchangebuyer // import "google.golang.org/api/adexchangebuyer/v1.4"

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

const apiId = "adexchangebuyer:v1.4"
const apiName = "adexchangebuyer"
const apiVersion = "v1.4"
const basePath = "https://www.googleapis.com/adexchangebuyer/v1.4/"

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
	s.BillingInfo = NewBillingInfoService(s)
	s.Budget = NewBudgetService(s)
	s.Creatives = NewCreativesService(s)
	s.Marketplacedeals = NewMarketplacedealsService(s)
	s.Marketplacenotes = NewMarketplacenotesService(s)
	s.Marketplaceoffers = NewMarketplaceoffersService(s)
	s.Marketplaceorders = NewMarketplaceordersService(s)
	s.PerformanceReport = NewPerformanceReportService(s)
	s.PretargetingConfig = NewPretargetingConfigService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Accounts *AccountsService

	BillingInfo *BillingInfoService

	Budget *BudgetService

	Creatives *CreativesService

	Marketplacedeals *MarketplacedealsService

	Marketplacenotes *MarketplacenotesService

	Marketplaceoffers *MarketplaceoffersService

	Marketplaceorders *MarketplaceordersService

	PerformanceReport *PerformanceReportService

	PretargetingConfig *PretargetingConfigService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAccountsService(s *Service) *AccountsService {
	rs := &AccountsService{s: s}
	return rs
}

type AccountsService struct {
	s *Service
}

func NewBillingInfoService(s *Service) *BillingInfoService {
	rs := &BillingInfoService{s: s}
	return rs
}

type BillingInfoService struct {
	s *Service
}

func NewBudgetService(s *Service) *BudgetService {
	rs := &BudgetService{s: s}
	return rs
}

type BudgetService struct {
	s *Service
}

func NewCreativesService(s *Service) *CreativesService {
	rs := &CreativesService{s: s}
	return rs
}

type CreativesService struct {
	s *Service
}

func NewMarketplacedealsService(s *Service) *MarketplacedealsService {
	rs := &MarketplacedealsService{s: s}
	return rs
}

type MarketplacedealsService struct {
	s *Service
}

func NewMarketplacenotesService(s *Service) *MarketplacenotesService {
	rs := &MarketplacenotesService{s: s}
	return rs
}

type MarketplacenotesService struct {
	s *Service
}

func NewMarketplaceoffersService(s *Service) *MarketplaceoffersService {
	rs := &MarketplaceoffersService{s: s}
	return rs
}

type MarketplaceoffersService struct {
	s *Service
}

func NewMarketplaceordersService(s *Service) *MarketplaceordersService {
	rs := &MarketplaceordersService{s: s}
	return rs
}

type MarketplaceordersService struct {
	s *Service
}

func NewPerformanceReportService(s *Service) *PerformanceReportService {
	rs := &PerformanceReportService{s: s}
	return rs
}

type PerformanceReportService struct {
	s *Service
}

func NewPretargetingConfigService(s *Service) *PretargetingConfigService {
	rs := &PretargetingConfigService{s: s}
	return rs
}

type PretargetingConfigService struct {
	s *Service
}

// Account: Configuration data for an Ad Exchange buyer account.
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

	// MaximumActiveCreatives: The maximum number of active creatives that
	// an account can have, where a creative is active if it was inserted or
	// bid with in the last 30 days. Please contact your technical account
	// manager if you need to change this.
	MaximumActiveCreatives int64 `json:"maximumActiveCreatives,omitempty"`

	// MaximumTotalQps: The sum of all bidderLocation.maximumQps values
	// cannot exceed this. Please contact your technical account manager if
	// you need to change this.
	MaximumTotalQps int64 `json:"maximumTotalQps,omitempty"`

	// NumberActiveCreatives: The number of creatives that this account
	// inserted or bid with in the last 30 days.
	NumberActiveCreatives int64 `json:"numberActiveCreatives,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BidderLocation") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Account) MarshalJSON() ([]byte, error) {
	type noMethod Account
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
	// - US_WEST
	Region string `json:"region,omitempty"`

	// Url: The URL to which the Ad Exchange will send bid requests.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaximumQps") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountBidderLocation) MarshalJSON() ([]byte, error) {
	type noMethod AccountBidderLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountsList: An account feed lists Ad Exchange buyer accounts that
// the user has access to. Each entry in the feed corresponds to a
// single buyer account.
type AccountsList struct {
	// Items: A list of accounts.
	Items []*Account `json:"items,omitempty"`

	// Kind: Resource type.
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
}

func (s *AccountsList) MarshalJSON() ([]byte, error) {
	type noMethod AccountsList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AddOrderDealsRequest struct {
	// Deals: The list of deals to add
	Deals []*MarketplaceDeal `json:"deals,omitempty"`

	// OrderRevisionNumber: The last known order revision number.
	OrderRevisionNumber int64 `json:"orderRevisionNumber,omitempty,string"`

	// UpdateAction: Indicates an optional action to take on the order
	UpdateAction string `json:"updateAction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Deals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AddOrderDealsRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddOrderDealsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AddOrderDealsResponse struct {
	// Deals: List of deals added (in the same order as passed in the
	// request)
	Deals []*MarketplaceDeal `json:"deals,omitempty"`

	// OrderRevisionNumber: The updated revision number for the order.
	OrderRevisionNumber int64 `json:"orderRevisionNumber,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AddOrderDealsResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddOrderDealsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AddOrderNotesRequest struct {
	// Notes: The list of notes to add.
	Notes []*MarketplaceNote `json:"notes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Notes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AddOrderNotesRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddOrderNotesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AddOrderNotesResponse struct {
	Notes []*MarketplaceNote `json:"notes,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Notes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AddOrderNotesResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddOrderNotesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BillingInfo: The configuration data for an Ad Exchange billing info.
type BillingInfo struct {
	// AccountId: Account id.
	AccountId int64 `json:"accountId,omitempty"`

	// AccountName: Account name.
	AccountName string `json:"accountName,omitempty"`

	// BillingId: A list of adgroup IDs associated with this particular
	// account. These IDs may show up as part of a realtime bidding
	// BidRequest, which indicates a bid request for this account.
	BillingId []string `json:"billingId,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BillingInfo) MarshalJSON() ([]byte, error) {
	type noMethod BillingInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BillingInfoList: A billing info feed lists Billing Info the Ad
// Exchange buyer account has access to. Each entry in the feed
// corresponds to a single billing info.
type BillingInfoList struct {
	// Items: A list of billing info relevant for your account.
	Items []*BillingInfo `json:"items,omitempty"`

	// Kind: Resource type.
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
}

func (s *BillingInfoList) MarshalJSON() ([]byte, error) {
	type noMethod BillingInfoList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Budget: The configuration data for Ad Exchange RTB - Budget API.
type Budget struct {
	// AccountId: The id of the account. This is required for get and update
	// requests.
	AccountId int64 `json:"accountId,omitempty,string"`

	// BillingId: The billing id to determine which adgroup to provide
	// budget information for. This is required for get and update requests.
	BillingId int64 `json:"billingId,omitempty,string"`

	// BudgetAmount: The daily budget amount in unit amount of the account
	// currency to apply for the billingId provided. This is required for
	// update requests.
	BudgetAmount int64 `json:"budgetAmount,omitempty,string"`

	// CurrencyCode: The currency code for the buyer. This cannot be altered
	// here.
	CurrencyCode string `json:"currencyCode,omitempty"`

	// Id: The unique id that describes this item.
	Id string `json:"id,omitempty"`

	// Kind: The kind of the resource, i.e. "adexchangebuyer#budget".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Budget) MarshalJSON() ([]byte, error) {
	type noMethod Budget
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Buyer struct {
	// AccountId: Adx account id of the buyer.
	AccountId string `json:"accountId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Buyer) MarshalJSON() ([]byte, error) {
	type noMethod Buyer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ContactInformation struct {
	// Email: Email address of the contact.
	Email string `json:"email,omitempty"`

	// Name: The name of the contact.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Email") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ContactInformation) MarshalJSON() ([]byte, error) {
	type noMethod ContactInformation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CreateOrdersRequest struct {
	// Orders: The list of orders to create.
	Orders []*MarketplaceOrder `json:"orders,omitempty"`

	WebPropertyCode string `json:"webPropertyCode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Orders") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreateOrdersRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateOrdersRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CreateOrdersResponse struct {
	// Orders: The list of orders successfully created.
	Orders []*MarketplaceOrder `json:"orders,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Orders") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreateOrdersResponse) MarshalJSON() ([]byte, error) {
	type noMethod CreateOrdersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Creative: A creative and its classification data.
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

	// ApiUploadTimestamp: The last upload timestamp of this creative if it
	// was uploaded via API. Read-only. The value of this field is
	// generated, and will be ignored for uploads. (formatted RFC 3339
	// timestamp).
	ApiUploadTimestamp string `json:"apiUploadTimestamp,omitempty"`

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

	// DealsStatus: Top-level deals status. Read-only. This field should not
	// be set in requests. If disapproved, an entry for
	// auctionType=DIRECT_DEALS (or ALL) in servingRestrictions will also
	// exist. Note that this may be nuanced with other contextual
	// restrictions, in which case it may be preferable to read from
	// servingRestrictions directly.
	DealsStatus string `json:"dealsStatus,omitempty"`

	// FilteringReasons: The filtering reasons for the creative. Read-only.
	// This field should not be set in requests.
	FilteringReasons *CreativeFilteringReasons `json:"filteringReasons,omitempty"`

	// Height: Ad height.
	Height int64 `json:"height,omitempty"`

	// ImpressionTrackingUrl: The set of urls to be called to record an
	// impression.
	ImpressionTrackingUrl []string `json:"impressionTrackingUrl,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// NativeAd: If nativeAd is set, HTMLSnippet and videoURL should not be
	// set.
	NativeAd *CreativeNativeAd `json:"nativeAd,omitempty"`

	// OpenAuctionStatus: Top-level open auction status. Read-only. This
	// field should not be set in requests. If disapproved, an entry for
	// auctionType=OPEN_AUCTION (or ALL) in servingRestrictions will also
	// exist. Note that this may be nuanced with other contextual
	// restrictions, in which case it may be preferable to read from
	// ServingRestrictions directly.
	OpenAuctionStatus string `json:"openAuctionStatus,omitempty"`

	// ProductCategories: Detected product categories, if any. Read-only.
	// This field should not be set in requests.
	ProductCategories []int64 `json:"productCategories,omitempty"`

	// RestrictedCategories: All restricted categories for the ads that may
	// be shown from this snippet.
	RestrictedCategories []int64 `json:"restrictedCategories,omitempty"`

	// SensitiveCategories: Detected sensitive categories, if any.
	// Read-only. This field should not be set in requests.
	SensitiveCategories []int64 `json:"sensitiveCategories,omitempty"`

	// ServingRestrictions: The granular status of this ad in specific
	// contexts. A context here relates to where something ultimately serves
	// (for example, a physical location, a platform, an HTTPS vs HTTP
	// request, or the type of auction). Read-only. This field should not be
	// set in requests.
	ServingRestrictions []*CreativeServingRestrictions `json:"servingRestrictions,omitempty"`

	// VendorType: All vendor types for the ads that may be shown from this
	// snippet.
	VendorType []int64 `json:"vendorType,omitempty"`

	// Version: The version for this creative. Read-only. This field should
	// not be set in requests.
	Version int64 `json:"version,omitempty"`

	// VideoURL: The url to fetch a video ad. If set, HTMLSnippet should not
	// be set.
	VideoURL string `json:"videoURL,omitempty"`

	// Width: Ad width.
	Width int64 `json:"width,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "HTMLSnippet") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Creative) MarshalJSON() ([]byte, error) {
	type noMethod Creative
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CreativeCorrections struct {
	// Details: Additional details about the correction.
	Details []string `json:"details,omitempty"`

	// Reason: The type of correction that was applied to the creative.
	Reason string `json:"reason,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeCorrections) MarshalJSON() ([]byte, error) {
	type noMethod CreativeCorrections
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreativeFilteringReasons: The filtering reasons for the creative.
// Read-only. This field should not be set in requests.
type CreativeFilteringReasons struct {
	// Date: The date in ISO 8601 format for the data. The data is collected
	// from 00:00:00 to 23:59:59 in PST.
	Date string `json:"date,omitempty"`

	// Reasons: The filtering reasons.
	Reasons []*CreativeFilteringReasonsReasons `json:"reasons,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Date") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeFilteringReasons) MarshalJSON() ([]byte, error) {
	type noMethod CreativeFilteringReasons
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CreativeFilteringReasonsReasons struct {
	// FilteringCount: The number of times the creative was filtered for the
	// status. The count is aggregated across all publishers on the
	// exchange.
	FilteringCount int64 `json:"filteringCount,omitempty,string"`

	// FilteringStatus: The filtering status code. Please refer to the
	// creative-status-codes.txt file for different statuses.
	FilteringStatus int64 `json:"filteringStatus,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FilteringCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeFilteringReasonsReasons) MarshalJSON() ([]byte, error) {
	type noMethod CreativeFilteringReasonsReasons
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreativeNativeAd: If nativeAd is set, HTMLSnippet and videoURL should
// not be set.
type CreativeNativeAd struct {
	Advertiser string `json:"advertiser,omitempty"`

	// AppIcon: The app icon, for app download ads.
	AppIcon *CreativeNativeAdAppIcon `json:"appIcon,omitempty"`

	// Body: A long description of the ad.
	Body string `json:"body,omitempty"`

	// CallToAction: A label for the button that the user is supposed to
	// click.
	CallToAction string `json:"callToAction,omitempty"`

	// ClickTrackingUrl: The URL to use for click tracking.
	ClickTrackingUrl string `json:"clickTrackingUrl,omitempty"`

	// Headline: A short title for the ad.
	Headline string `json:"headline,omitempty"`

	// Image: A large image.
	Image *CreativeNativeAdImage `json:"image,omitempty"`

	// ImpressionTrackingUrl: The URLs are called when the impression is
	// rendered.
	ImpressionTrackingUrl []string `json:"impressionTrackingUrl,omitempty"`

	// Logo: A smaller image, for the advertiser logo.
	Logo *CreativeNativeAdLogo `json:"logo,omitempty"`

	// Price: The price of the promoted app including the currency info.
	Price string `json:"price,omitempty"`

	// StarRating: The app rating in the app store. Must be in the range
	// [0-5].
	StarRating float64 `json:"starRating,omitempty"`

	// Store: The URL to the app store to purchase/download the promoted
	// app.
	Store string `json:"store,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Advertiser") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeNativeAd) MarshalJSON() ([]byte, error) {
	type noMethod CreativeNativeAd
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreativeNativeAdAppIcon: The app icon, for app download ads.
type CreativeNativeAdAppIcon struct {
	Height int64 `json:"height,omitempty"`

	Url string `json:"url,omitempty"`

	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeNativeAdAppIcon) MarshalJSON() ([]byte, error) {
	type noMethod CreativeNativeAdAppIcon
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreativeNativeAdImage: A large image.
type CreativeNativeAdImage struct {
	Height int64 `json:"height,omitempty"`

	Url string `json:"url,omitempty"`

	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeNativeAdImage) MarshalJSON() ([]byte, error) {
	type noMethod CreativeNativeAdImage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreativeNativeAdLogo: A smaller image, for the advertiser logo.
type CreativeNativeAdLogo struct {
	Height int64 `json:"height,omitempty"`

	Url string `json:"url,omitempty"`

	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeNativeAdLogo) MarshalJSON() ([]byte, error) {
	type noMethod CreativeNativeAdLogo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CreativeServingRestrictions struct {
	// Contexts: All known contexts/restrictions.
	Contexts []*CreativeServingRestrictionsContexts `json:"contexts,omitempty"`

	// DisapprovalReasons: The reasons for disapproval within this
	// restriction, if any. Note that not all disapproval reasons may be
	// categorized, so it is possible for the creative to have a status of
	// DISAPPROVED or CONDITIONALLY_APPROVED with an empty list for
	// disapproval_reasons. In this case, please reach out to your TAM to
	// help debug the issue.
	DisapprovalReasons []*CreativeServingRestrictionsDisapprovalReasons `json:"disapprovalReasons,omitempty"`

	// Reason: Why the creative is ineligible to serve in this context
	// (e.g., it has been explicitly disapproved or is pending review).
	Reason string `json:"reason,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Contexts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeServingRestrictions) MarshalJSON() ([]byte, error) {
	type noMethod CreativeServingRestrictions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CreativeServingRestrictionsContexts struct {
	// AuctionType: Only set when contextType=AUCTION_TYPE. Represents the
	// auction types this restriction applies to.
	AuctionType []string `json:"auctionType,omitempty"`

	// ContextType: The type of context (e.g., location, platform, auction
	// type, SSL-ness).
	ContextType string `json:"contextType,omitempty"`

	// GeoCriteriaId: Only set when contextType=LOCATION. Represents the geo
	// criterias this restriction applies to.
	GeoCriteriaId []int64 `json:"geoCriteriaId,omitempty"`

	// Platform: Only set when contextType=PLATFORM. Represents the
	// platforms this restriction applies to.
	Platform []string `json:"platform,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuctionType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeServingRestrictionsContexts) MarshalJSON() ([]byte, error) {
	type noMethod CreativeServingRestrictionsContexts
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CreativeServingRestrictionsDisapprovalReasons struct {
	// Details: Additional details about the reason for disapproval.
	Details []string `json:"details,omitempty"`

	// Reason: The categorized reason for disapproval.
	Reason string `json:"reason,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreativeServingRestrictionsDisapprovalReasons) MarshalJSON() ([]byte, error) {
	type noMethod CreativeServingRestrictionsDisapprovalReasons
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreativesList: The creatives feed lists the active creatives for the
// Ad Exchange buyer accounts that the user has access to. Each entry in
// the feed corresponds to a single creative.
type CreativesList struct {
	// Items: A list of creatives.
	Items []*Creative `json:"items,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through creatives. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`

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
}

func (s *CreativesList) MarshalJSON() ([]byte, error) {
	type noMethod CreativesList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DealTerms struct {
	// BrandingType: Visibilty of the URL in bid requests.
	BrandingType string `json:"brandingType,omitempty"`

	// Description: Description for the proposed terms of the deal.
	Description string `json:"description,omitempty"`

	// EstimatedGrossSpend: Non-binding estimate of the estimated gross
	// spend for this deal Can be set by buyer or seller.
	EstimatedGrossSpend *Price `json:"estimatedGrossSpend,omitempty"`

	// EstimatedImpressionsPerDay: Non-binding estimate of the impressions
	// served per day Can be set by buyer or seller.
	EstimatedImpressionsPerDay int64 `json:"estimatedImpressionsPerDay,omitempty,string"`

	// GuaranteedFixedPriceTerms: The terms for guaranteed fixed price
	// deals.
	GuaranteedFixedPriceTerms *DealTermsGuaranteedFixedPriceTerms `json:"guaranteedFixedPriceTerms,omitempty"`

	// NonGuaranteedAuctionTerms: The terms for non-guaranteed auction
	// deals.
	NonGuaranteedAuctionTerms *DealTermsNonGuaranteedAuctionTerms `json:"nonGuaranteedAuctionTerms,omitempty"`

	// NonGuaranteedFixedPriceTerms: The terms for non-guaranteed fixed
	// price deals.
	NonGuaranteedFixedPriceTerms *DealTermsNonGuaranteedFixedPriceTerms `json:"nonGuaranteedFixedPriceTerms,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BrandingType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DealTerms) MarshalJSON() ([]byte, error) {
	type noMethod DealTerms
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DealTermsGuaranteedFixedPriceTerms struct {
	// FixedPrices: Fixed price for the specified buyer.
	FixedPrices []*PricePerBuyer `json:"fixedPrices,omitempty"`

	// GuaranteedImpressions: Guaranteed impressions as a percentage. This
	// is the percentage of guaranteed looks that the buyer is guaranteeing
	// to buy.
	GuaranteedImpressions int64 `json:"guaranteedImpressions,omitempty,string"`

	// GuaranteedLooks: Count of guaranteed looks. Required for deal,
	// optional for offer.
	GuaranteedLooks int64 `json:"guaranteedLooks,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "FixedPrices") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DealTermsGuaranteedFixedPriceTerms) MarshalJSON() ([]byte, error) {
	type noMethod DealTermsGuaranteedFixedPriceTerms
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DealTermsNonGuaranteedAuctionTerms struct {
	// PrivateAuctionId: Id of the corresponding private auction.
	PrivateAuctionId string `json:"privateAuctionId,omitempty"`

	// ReservePricePerBuyers: Reserve price for the specified buyer.
	ReservePricePerBuyers []*PricePerBuyer `json:"reservePricePerBuyers,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PrivateAuctionId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DealTermsNonGuaranteedAuctionTerms) MarshalJSON() ([]byte, error) {
	type noMethod DealTermsNonGuaranteedAuctionTerms
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DealTermsNonGuaranteedFixedPriceTerms struct {
	// FixedPrices: Fixed price for the specified buyer.
	FixedPrices []*PricePerBuyer `json:"fixedPrices,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FixedPrices") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DealTermsNonGuaranteedFixedPriceTerms) MarshalJSON() ([]byte, error) {
	type noMethod DealTermsNonGuaranteedFixedPriceTerms
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DeleteOrderDealsRequest struct {
	// DealIds: List of deals to delete for a given order
	DealIds []string `json:"dealIds,omitempty"`

	// OrderRevisionNumber: The last known order revision number.
	OrderRevisionNumber int64 `json:"orderRevisionNumber,omitempty,string"`

	UpdateAction string `json:"updateAction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DealIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeleteOrderDealsRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteOrderDealsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DeleteOrderDealsResponse struct {
	// Deals: List of deals deleted (in the same order as passed in the
	// request)
	Deals []*MarketplaceDeal `json:"deals,omitempty"`

	// OrderRevisionNumber: The updated revision number for the order.
	OrderRevisionNumber int64 `json:"orderRevisionNumber,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeleteOrderDealsResponse) MarshalJSON() ([]byte, error) {
	type noMethod DeleteOrderDealsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DeliveryControl struct {
	CreativeBlockingLevel string `json:"creativeBlockingLevel,omitempty"`

	DeliveryRateType string `json:"deliveryRateType,omitempty"`

	FrequencyCaps []*DeliveryControlFrequencyCap `json:"frequencyCaps,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "CreativeBlockingLevel") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeliveryControl) MarshalJSON() ([]byte, error) {
	type noMethod DeliveryControl
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DeliveryControlFrequencyCap struct {
	MaxImpressions int64 `json:"maxImpressions,omitempty"`

	NumTimeUnits int64 `json:"numTimeUnits,omitempty"`

	TimeUnitType string `json:"timeUnitType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxImpressions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeliveryControlFrequencyCap) MarshalJSON() ([]byte, error) {
	type noMethod DeliveryControlFrequencyCap
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type EditAllOrderDealsRequest struct {
	// Deals: List of deals to edit. Service may perform 3 different
	// operations based on comparison of deals in this list vs deals already
	// persisted in database: 1. Add new deal to order If a deal in this
	// list does not exist in the order, the service will create a new deal
	// and add it to the order. Validation will follow AddOrderDealsRequest.
	// 2. Update existing deal in the order If a deal in this list already
	// exist in the order, the service will update that existing deal to
	// this new deal in the request. Validation will follow
	// UpdateOrderDealsRequest. 3. Delete deals from the order (just need
	// the id) If a existing deal in the order is not present in this list,
	// the service will delete that deal from the order. Validation will
	// follow DeleteOrderDealsRequest.
	Deals []*MarketplaceDeal `json:"deals,omitempty"`

	// Order: If specified, also updates the order in the batch transaction.
	// This is useful when the order and the deals need to be updated in one
	// transaction.
	Order *MarketplaceOrder `json:"order,omitempty"`

	// OrderRevisionNumber: The last known revision number for the order.
	OrderRevisionNumber int64 `json:"orderRevisionNumber,omitempty,string"`

	// UpdateAction: Indicates an optional action to take on the order
	UpdateAction string `json:"updateAction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Deals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EditAllOrderDealsRequest) MarshalJSON() ([]byte, error) {
	type noMethod EditAllOrderDealsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type EditAllOrderDealsResponse struct {
	// Deals: List of all deals in the order after edit.
	Deals []*MarketplaceDeal `json:"deals,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EditAllOrderDealsResponse) MarshalJSON() ([]byte, error) {
	type noMethod EditAllOrderDealsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GetOffersResponse struct {
	// Offers: The returned list of offers.
	Offers []*MarketplaceOffer `json:"offers,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Offers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetOffersResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetOffersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GetOrderDealsResponse struct {
	// Deals: List of deals for the order
	Deals []*MarketplaceDeal `json:"deals,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetOrderDealsResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetOrderDealsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GetOrderNotesResponse struct {
	// Notes: The list of matching notes.
	Notes []*MarketplaceNote `json:"notes,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Notes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetOrderNotesResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetOrderNotesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GetOrdersResponse struct {
	// Orders: The list of matching orders.
	Orders []*MarketplaceOrder `json:"orders,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Orders") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetOrdersResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetOrdersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// MarketplaceDeal: An order can contain multiple deals. A deal contains
// the terms and targeting information that is used for serving.
type MarketplaceDeal struct {
	// BuyerPrivateData: Buyer private data (hidden from seller).
	BuyerPrivateData *PrivateData `json:"buyerPrivateData,omitempty"`

	// CreationTimeMs: The time (ms since epoch) of the deal creation.
	// (readonly)
	CreationTimeMs int64 `json:"creationTimeMs,omitempty,string"`

	// CreativePreApprovalPolicy: Specifies the creative pre-approval policy
	// (buyer-readonly)
	CreativePreApprovalPolicy string `json:"creativePreApprovalPolicy,omitempty"`

	// DealId: A unique deal=id for the deal (readonly).
	DealId string `json:"dealId,omitempty"`

	// DeliveryControl: The set of fields around delivery control that are
	// interesting for a buyer to see but are non-negotiable. These are set
	// by the publisher. This message is assigned an id of 100 since some
	// day we would want to model this as a protobuf extension.
	DeliveryControl *DeliveryControl `json:"deliveryControl,omitempty"`

	// ExternalDealId: The external deal id assigned to this deal once the
	// deal is finalized. This is the deal-id that shows up in
	// serving/reporting etc. (readonly)
	ExternalDealId string `json:"externalDealId,omitempty"`

	// FlightEndTimeMs: Proposed flight end time of the deal (ms since
	// epoch) This will generally be stored in a granularity of a second.
	// (updatable)
	FlightEndTimeMs int64 `json:"flightEndTimeMs,omitempty,string"`

	// FlightStartTimeMs: Proposed flight start time of the deal (ms since
	// epoch) This will generally be stored in a granularity of a second.
	// (updatable)
	FlightStartTimeMs int64 `json:"flightStartTimeMs,omitempty,string"`

	// InventoryDescription: Description for the deal terms. (updatable)
	InventoryDescription string `json:"inventoryDescription,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "adexchangebuyer#marketplaceDeal".
	Kind string `json:"kind,omitempty"`

	// LastUpdateTimeMs: The time (ms since epoch) when the deal was last
	// updated. (readonly)
	LastUpdateTimeMs int64 `json:"lastUpdateTimeMs,omitempty,string"`

	// Name: The name of the deal. (updatable)
	Name string `json:"name,omitempty"`

	// OfferId: The offer-id from which this deal was created. (readonly,
	// except on create)
	OfferId string `json:"offerId,omitempty"`

	// OfferRevisionNumber: The revision number of the offer that the deal
	// was created from (readonly, except on create)
	OfferRevisionNumber int64 `json:"offerRevisionNumber,omitempty,string"`

	OrderId string `json:"orderId,omitempty"`

	// SellerContacts: Optional Seller contact information for the deal
	// (buyer-readonly)
	SellerContacts []*ContactInformation `json:"sellerContacts,omitempty"`

	// SharedTargetings: The shared targeting visible to buyers and sellers.
	// (updatable)
	SharedTargetings []*SharedTargeting `json:"sharedTargetings,omitempty"`

	// SyndicationProduct: The syndication product associated with the deal.
	// (readonly, except on create)
	SyndicationProduct string `json:"syndicationProduct,omitempty"`

	// Terms: The negotiable terms of the deal. (updatable)
	Terms *DealTerms `json:"terms,omitempty"`

	WebPropertyCode string `json:"webPropertyCode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BuyerPrivateData") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MarketplaceDeal) MarshalJSON() ([]byte, error) {
	type noMethod MarketplaceDeal
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type MarketplaceDealParty struct {
	// Buyer: The buyer/seller associated with the deal. One of buyer/seller
	// is specified for a deal-party.
	Buyer *Buyer `json:"buyer,omitempty"`

	// Seller: The buyer/seller associated with the deal. One of
	// buyer/seller is specified for a deal party.
	Seller *Seller `json:"seller,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Buyer") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MarketplaceDealParty) MarshalJSON() ([]byte, error) {
	type noMethod MarketplaceDealParty
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type MarketplaceLabel struct {
	// AccountId: The accountId of the party that created the label.
	AccountId string `json:"accountId,omitempty"`

	// CreateTimeMs: The creation time (in ms since epoch) for the label.
	CreateTimeMs int64 `json:"createTimeMs,omitempty,string"`

	// DeprecatedMarketplaceDealParty: Information about the party that
	// created the label.
	DeprecatedMarketplaceDealParty *MarketplaceDealParty `json:"deprecatedMarketplaceDealParty,omitempty"`

	// Label: The label to use.
	Label string `json:"label,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MarketplaceLabel) MarshalJSON() ([]byte, error) {
	type noMethod MarketplaceLabel
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// MarketplaceNote: An order is associated with a bunch of notes which
// may optionally be associated with a deal and/or revision number.
type MarketplaceNote struct {
	// CreatorRole: The role of the person (buyer/seller) creating the note.
	// (readonly)
	CreatorRole string `json:"creatorRole,omitempty"`

	// DealId: Notes can optionally be associated with a deal. (readonly,
	// except on create)
	DealId string `json:"dealId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "adexchangebuyer#marketplaceNote".
	Kind string `json:"kind,omitempty"`

	// Note: The actual note to attach. (readonly, except on create)
	Note string `json:"note,omitempty"`

	// NoteId: The unique id for the note. (readonly)
	NoteId string `json:"noteId,omitempty"`

	// OrderId: The order_id that a note is attached to. (readonly)
	OrderId string `json:"orderId,omitempty"`

	// OrderRevisionNumber: If the note is associated with an order revision
	// number, then store that here. (readonly, except on create)
	OrderRevisionNumber int64 `json:"orderRevisionNumber,omitempty,string"`

	// TimestampMs: The timestamp (ms since epoch) that this note was
	// created. (readonly)
	TimestampMs int64 `json:"timestampMs,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "CreatorRole") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MarketplaceNote) MarshalJSON() ([]byte, error) {
	type noMethod MarketplaceNote
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// MarketplaceOffer: An offer is segment of inventory that a seller
// wishes to sell. It is associated with certain terms and targeting
// information which helps buyer know more about the inventory. Each
// field in an order can have one of the following setting:
//
// (readonly) - It is an error to try and set this field.
// (buyer-readonly) - Only the seller can set this field.
// (seller-readonly) - Only the buyer can set this field. (updatable) -
// The field is updatable at all times by either buyer or the seller.
type MarketplaceOffer struct {
	// CreationTimeMs: Creation time in ms. since epoch (readonly)
	CreationTimeMs int64 `json:"creationTimeMs,omitempty,string"`

	// CreatorContacts: Optional contact information for the creator of this
	// offer. (buyer-readonly)
	CreatorContacts []*ContactInformation `json:"creatorContacts,omitempty"`

	// FlightEndTimeMs: The proposed end time for the deal (ms since epoch)
	// (buyer-readonly)
	FlightEndTimeMs int64 `json:"flightEndTimeMs,omitempty,string"`

	// FlightStartTimeMs: Inventory availability dates. (times are in ms
	// since epoch) The granularity is generally in the order of seconds.
	// (buyer-readonly)
	FlightStartTimeMs int64 `json:"flightStartTimeMs,omitempty,string"`

	// HasCreatorSignedOff: If the creator has already signed off on the
	// offer, then the buyer can finalize the deal by accepting the offer as
	// is. When copying to an order, if any of the terms are changed, then
	// auto_finalize is automatically set to false.
	HasCreatorSignedOff bool `json:"hasCreatorSignedOff,omitempty"`

	// InventorySource: What exchange will provide this inventory (readonly,
	// except on create).
	InventorySource string `json:"inventorySource,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "adexchangebuyer#marketplaceOffer".
	Kind string `json:"kind,omitempty"`

	// Labels: Optional List of labels for the offer (optional,
	// buyer-readonly).
	Labels []*MarketplaceLabel `json:"labels,omitempty"`

	// LastUpdateTimeMs: Time of last update in ms. since epoch (readonly)
	LastUpdateTimeMs int64 `json:"lastUpdateTimeMs,omitempty,string"`

	// Name: The name for this offer as set by the seller. (buyer-readonly)
	Name string `json:"name,omitempty"`

	// OfferId: The unique id for the offer (readonly)
	OfferId string `json:"offerId,omitempty"`

	// RevisionNumber: The revision number of the offer. (readonly)
	RevisionNumber int64 `json:"revisionNumber,omitempty,string"`

	// Seller: Information about the seller that created this offer
	// (readonly, except on create)
	Seller *Seller `json:"seller,omitempty"`

	// SharedTargetings: Targeting that is shared between the buyer and the
	// seller. Each targeting criteria has a specified key and for each key
	// there is a list of inclusion value or exclusion values.
	// (buyer-readonly)
	SharedTargetings []*SharedTargeting `json:"sharedTargetings,omitempty"`

	// State: The state of the offer. (buyer-readonly)
	State string `json:"state,omitempty"`

	// SyndicationProduct: The syndication product associated with the deal.
	// (readonly, except on create)
	SyndicationProduct string `json:"syndicationProduct,omitempty"`

	// Terms: The negotiable terms of the deal (buyer-readonly)
	Terms *DealTerms `json:"terms,omitempty"`

	WebPropertyCode string `json:"webPropertyCode,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationTimeMs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MarketplaceOffer) MarshalJSON() ([]byte, error) {
	type noMethod MarketplaceOffer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// MarketplaceOrder: Represents an order in the marketplace. An order is
// the unit of negotiation between a seller and a buyer and contains
// deals which are served. Each field in an order can have one of the
// following setting:
//
// (readonly) - It is an error to try and set this field.
// (buyer-readonly) - Only the seller can set this field.
// (seller-readonly) - Only the buyer can set this field. (updatable) -
// The field is updatable at all times by either buyer or the seller.
type MarketplaceOrder struct {
	// BilledBuyer: Reference to the buyer that will get billed for this
	// order. (readonly)
	BilledBuyer *Buyer `json:"billedBuyer,omitempty"`

	// Buyer: Reference to the buyer on the order. (readonly, except on
	// create)
	Buyer *Buyer `json:"buyer,omitempty"`

	// BuyerContacts: Optional contact information fort the buyer.
	// (seller-readonly)
	BuyerContacts []*ContactInformation `json:"buyerContacts,omitempty"`

	// BuyerPrivateData: Private data for buyer. (hidden from seller).
	BuyerPrivateData *PrivateData `json:"buyerPrivateData,omitempty"`

	// HasBuyerSignedOff: When an order is in an accepted state, indicates
	// whether the buyer has signed off Once both sides have signed off on a
	// deal, the order can be finalized by the seller. (seller-readonly)
	HasBuyerSignedOff bool `json:"hasBuyerSignedOff,omitempty"`

	// HasSellerSignedOff: When an order is in an accepted state, indicates
	// whether the buyer has signed off Once both sides have signed off on a
	// deal, the order can be finalized by the seller. (buyer-readonly)
	HasSellerSignedOff bool `json:"hasSellerSignedOff,omitempty"`

	// InventorySource: What exchange will provide this inventory (readonly,
	// except on create).
	InventorySource string `json:"inventorySource,omitempty"`

	// IsRenegotiating: True if the order is being renegotiated (readonly).
	IsRenegotiating bool `json:"isRenegotiating,omitempty"`

	// IsSetupComplete: True, if the buyside inventory setup is complete for
	// this order. (readonly)
	IsSetupComplete bool `json:"isSetupComplete,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "adexchangebuyer#marketplaceOrder".
	Kind string `json:"kind,omitempty"`

	// Labels: List of labels associated with the order. (readonly)
	Labels []*MarketplaceLabel `json:"labels,omitempty"`

	// LastUpdaterOrCommentorRole: The role of the last user that either
	// updated the order or left a comment. (readonly)
	LastUpdaterOrCommentorRole string `json:"lastUpdaterOrCommentorRole,omitempty"`

	LastUpdaterRole string `json:"lastUpdaterRole,omitempty"`

	// Name: The name for the order (updatable)
	Name string `json:"name,omitempty"`

	// OrderId: The unique id of the order. (readonly).
	OrderId string `json:"orderId,omitempty"`

	// OrderState: The current state of the order. (readonly)
	OrderState string `json:"orderState,omitempty"`

	// OriginatorRole: Indicates whether the buyer/seller created the
	// offer.(readonly)
	OriginatorRole string `json:"originatorRole,omitempty"`

	// RevisionNumber: The revision number for the order (readonly).
	RevisionNumber int64 `json:"revisionNumber,omitempty,string"`

	// RevisionTimeMs: The time (ms since epoch) when the order was last
	// revised (readonly).
	RevisionTimeMs int64 `json:"revisionTimeMs,omitempty,string"`

	// Seller: Reference to the seller on the order. (readonly, except on
	// create)
	Seller *Seller `json:"seller,omitempty"`

	// SellerContacts: Optional contact information for the seller
	// (buyer-readonly).
	SellerContacts []*ContactInformation `json:"sellerContacts,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BilledBuyer") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MarketplaceOrder) MarshalJSON() ([]byte, error) {
	type noMethod MarketplaceOrder
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PerformanceReport: The configuration data for an Ad Exchange
// performance report list.
type PerformanceReport struct {
	// BidRate: The number of bid responses with an ad.
	BidRate float64 `json:"bidRate,omitempty"`

	// BidRequestRate: The number of bid requests sent to your bidder.
	BidRequestRate float64 `json:"bidRequestRate,omitempty"`

	// CalloutStatusRate: Rate of various prefiltering statuses per match.
	// Please refer to the callout-status-codes.txt file for different
	// statuses.
	CalloutStatusRate []interface{} `json:"calloutStatusRate,omitempty"`

	// CookieMatcherStatusRate: Average QPS for cookie matcher operations.
	CookieMatcherStatusRate []interface{} `json:"cookieMatcherStatusRate,omitempty"`

	// CreativeStatusRate: Rate of ads with a given status. Please refer to
	// the creative-status-codes.txt file for different statuses.
	CreativeStatusRate []interface{} `json:"creativeStatusRate,omitempty"`

	// FilteredBidRate: The number of bid responses that were filtered due
	// to a policy violation or other errors.
	FilteredBidRate float64 `json:"filteredBidRate,omitempty"`

	// HostedMatchStatusRate: Average QPS for hosted match operations.
	HostedMatchStatusRate []interface{} `json:"hostedMatchStatusRate,omitempty"`

	// InventoryMatchRate: The number of potential queries based on your
	// pretargeting settings.
	InventoryMatchRate float64 `json:"inventoryMatchRate,omitempty"`

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

	// SuccessfulRequestRate: The number of properly formed bid responses
	// received by our servers within the deadline.
	SuccessfulRequestRate float64 `json:"successfulRequestRate,omitempty"`

	// Timestamp: The unix timestamp of the starting time of this
	// performance data.
	Timestamp int64 `json:"timestamp,omitempty,string"`

	// UnsuccessfulRequestRate: The number of bid responses that were
	// unsuccessful due to timeouts, incorrect formatting, etc.
	UnsuccessfulRequestRate float64 `json:"unsuccessfulRequestRate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BidRate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PerformanceReport) MarshalJSON() ([]byte, error) {
	type noMethod PerformanceReport
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PerformanceReportList: The configuration data for an Ad Exchange
// performance report list.
// https://sites.google.com/a/google.com/adx-integration/Home/engineering/binary-releases/rtb-api-release
// https://cs.corp.google.com/#piper///depot/google3/contentads/adx/tools/rtb_api/adxrtb.py
type PerformanceReportList struct {
	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// PerformanceReport: A list of performance reports relevant for the
	// account.
	PerformanceReport []*PerformanceReport `json:"performanceReport,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PerformanceReportList) MarshalJSON() ([]byte, error) {
	type noMethod PerformanceReportList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PretargetingConfig struct {
	// BillingId: The id for billing purposes, provided for reference. Leave
	// this field blank for insert requests; the id will be generated
	// automatically.
	BillingId int64 `json:"billingId,omitempty,string"`

	// ConfigId: The config id; generated automatically. Leave this field
	// blank for insert requests.
	ConfigId int64 `json:"configId,omitempty,string"`

	// ConfigName: The name of the config. Must be unique. Required for all
	// requests.
	ConfigName string `json:"configName,omitempty"`

	// CreativeType: List must contain exactly one of
	// PRETARGETING_CREATIVE_TYPE_HTML or PRETARGETING_CREATIVE_TYPE_VIDEO.
	CreativeType []string `json:"creativeType,omitempty"`

	// Dimensions: Requests which allow one of these (width, height) pairs
	// will match. All pairs must be supported ad dimensions.
	Dimensions []*PretargetingConfigDimensions `json:"dimensions,omitempty"`

	// ExcludedContentLabels: Requests with any of these content labels will
	// not match. Values are from content-labels.txt in the downloadable
	// files section.
	ExcludedContentLabels googleapi.Int64s `json:"excludedContentLabels,omitempty"`

	// ExcludedGeoCriteriaIds: Requests containing any of these geo criteria
	// ids will not match.
	ExcludedGeoCriteriaIds googleapi.Int64s `json:"excludedGeoCriteriaIds,omitempty"`

	// ExcludedPlacements: Requests containing any of these placements will
	// not match.
	ExcludedPlacements []*PretargetingConfigExcludedPlacements `json:"excludedPlacements,omitempty"`

	// ExcludedUserLists: Requests containing any of these users list ids
	// will not match.
	ExcludedUserLists googleapi.Int64s `json:"excludedUserLists,omitempty"`

	// ExcludedVerticals: Requests containing any of these vertical ids will
	// not match. Values are from the publisher-verticals.txt file in the
	// downloadable files section.
	ExcludedVerticals googleapi.Int64s `json:"excludedVerticals,omitempty"`

	// GeoCriteriaIds: Requests containing any of these geo criteria ids
	// will match.
	GeoCriteriaIds googleapi.Int64s `json:"geoCriteriaIds,omitempty"`

	// IsActive: Whether this config is active. Required for all requests.
	IsActive bool `json:"isActive,omitempty"`

	// Kind: The kind of the resource, i.e.
	// "adexchangebuyer#pretargetingConfig".
	Kind string `json:"kind,omitempty"`

	// Languages: Request containing any of these language codes will match.
	Languages []string `json:"languages,omitempty"`

	// MobileCarriers: Requests containing any of these mobile carrier ids
	// will match. Values are from mobile-carriers.csv in the downloadable
	// files section.
	MobileCarriers googleapi.Int64s `json:"mobileCarriers,omitempty"`

	// MobileDevices: Requests containing any of these mobile device ids
	// will match. Values are from mobile-devices.csv in the downloadable
	// files section.
	MobileDevices googleapi.Int64s `json:"mobileDevices,omitempty"`

	// MobileOperatingSystemVersions: Requests containing any of these
	// mobile operating system version ids will match. Values are from
	// mobile-os.csv in the downloadable files section.
	MobileOperatingSystemVersions googleapi.Int64s `json:"mobileOperatingSystemVersions,omitempty"`

	// Placements: Requests containing any of these placements will match.
	Placements []*PretargetingConfigPlacements `json:"placements,omitempty"`

	// Platforms: Requests matching any of these platforms will match.
	// Possible values are PRETARGETING_PLATFORM_MOBILE,
	// PRETARGETING_PLATFORM_DESKTOP, and PRETARGETING_PLATFORM_TABLET.
	Platforms []string `json:"platforms,omitempty"`

	// SupportedCreativeAttributes: Creative attributes should be declared
	// here if all creatives corresponding to this pretargeting
	// configuration have that creative attribute. Values are from
	// pretargetable-creative-attributes.txt in the downloadable files
	// section.
	SupportedCreativeAttributes googleapi.Int64s `json:"supportedCreativeAttributes,omitempty"`

	// UserLists: Requests containing any of these user list ids will match.
	UserLists googleapi.Int64s `json:"userLists,omitempty"`

	// VendorTypes: Requests that allow any of these vendor ids will match.
	// Values are from vendors.txt in the downloadable files section.
	VendorTypes googleapi.Int64s `json:"vendorTypes,omitempty"`

	// Verticals: Requests containing any of these vertical ids will match.
	Verticals googleapi.Int64s `json:"verticals,omitempty"`

	// VideoPlayerSizes: Video requests satisfying any of these player size
	// constraints will match.
	VideoPlayerSizes []*PretargetingConfigVideoPlayerSizes `json:"videoPlayerSizes,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BillingId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PretargetingConfig) MarshalJSON() ([]byte, error) {
	type noMethod PretargetingConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PretargetingConfigDimensions struct {
	// Height: Height in pixels.
	Height int64 `json:"height,omitempty,string"`

	// Width: Width in pixels.
	Width int64 `json:"width,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PretargetingConfigDimensions) MarshalJSON() ([]byte, error) {
	type noMethod PretargetingConfigDimensions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PretargetingConfigExcludedPlacements struct {
	// Token: The value of the placement. Interpretation depends on the
	// placement type, e.g. URL for a site placement, channel name for a
	// channel placement, app id for a mobile app placement.
	Token string `json:"token,omitempty"`

	// Type: The type of the placement.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Token") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PretargetingConfigExcludedPlacements) MarshalJSON() ([]byte, error) {
	type noMethod PretargetingConfigExcludedPlacements
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PretargetingConfigPlacements struct {
	// Token: The value of the placement. Interpretation depends on the
	// placement type, e.g. URL for a site placement, channel name for a
	// channel placement, app id for a mobile app placement.
	Token string `json:"token,omitempty"`

	// Type: The type of the placement.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Token") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PretargetingConfigPlacements) MarshalJSON() ([]byte, error) {
	type noMethod PretargetingConfigPlacements
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PretargetingConfigVideoPlayerSizes struct {
	// AspectRatio: The type of aspect ratio. Leave this field blank to
	// match all aspect ratios.
	AspectRatio string `json:"aspectRatio,omitempty"`

	// MinHeight: The minimum player height in pixels. Leave this field
	// blank to match any player height.
	MinHeight int64 `json:"minHeight,omitempty,string"`

	// MinWidth: The minimum player width in pixels. Leave this field blank
	// to match any player width.
	MinWidth int64 `json:"minWidth,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "AspectRatio") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PretargetingConfigVideoPlayerSizes) MarshalJSON() ([]byte, error) {
	type noMethod PretargetingConfigVideoPlayerSizes
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PretargetingConfigList struct {
	// Items: A list of pretargeting configs
	Items []*PretargetingConfig `json:"items,omitempty"`

	// Kind: Resource type.
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
}

func (s *PretargetingConfigList) MarshalJSON() ([]byte, error) {
	type noMethod PretargetingConfigList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Price struct {
	// AmountMicros: The price value in micros.
	AmountMicros float64 `json:"amountMicros,omitempty"`

	// CurrencyCode: The currency code for the price.
	CurrencyCode string `json:"currencyCode,omitempty"`

	// PricingType: The pricing type for the deal/offer.
	PricingType string `json:"pricingType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AmountMicros") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Price) MarshalJSON() ([]byte, error) {
	type noMethod Price
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PricePerBuyer: Used to specify pricing rules for buyers/advertisers.
// Each PricePerBuyer in an offer can become [0,1] deals. To check if
// there is a PricePerBuyer for a particular buyer or buyer/advertiser
// pair, we look for the most specific matching rule - we first look for
// a rule matching the buyer and advertiser, next a rule with the buyer
// but an empty advertiser list, and otherwise look for a matching rule
// where no buyer is set.
type PricePerBuyer struct {
	// Buyer: The buyer who will pay this price. If unset, all buyers can
	// pay this price (if the advertisers match, and there's no more
	// specific rule matching the buyer).
	Buyer *Buyer `json:"buyer,omitempty"`

	// Price: The specified price
	Price *Price `json:"price,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Buyer") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PricePerBuyer) MarshalJSON() ([]byte, error) {
	type noMethod PricePerBuyer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type PrivateData struct {
	ReferenceId string `json:"referenceId,omitempty"`

	ReferencePayload string `json:"referencePayload,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReferenceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PrivateData) MarshalJSON() ([]byte, error) {
	type noMethod PrivateData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Seller struct {
	// AccountId: The unique id for the seller. The seller fills in this
	// field. The seller account id is then available to buyer in the offer.
	AccountId string `json:"accountId,omitempty"`

	// SubAccountId: Optional sub-account id for the seller.
	SubAccountId string `json:"subAccountId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Seller) MarshalJSON() ([]byte, error) {
	type noMethod Seller
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type SharedTargeting struct {
	// Exclusions: The list of values to exclude from targeting.
	Exclusions []*TargetingValue `json:"exclusions,omitempty"`

	// Inclusions: The list of value to include as part of the targeting.
	Inclusions []*TargetingValue `json:"inclusions,omitempty"`

	// Key: The key representing the shared targeting criterion.
	Key string `json:"key,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Exclusions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SharedTargeting) MarshalJSON() ([]byte, error) {
	type noMethod SharedTargeting
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TargetingValue struct {
	// CreativeSizeValue: The creative size value to exclude/include.
	CreativeSizeValue *TargetingValueCreativeSize `json:"creativeSizeValue,omitempty"`

	// DayPartTargetingValue: The daypart targeting to include / exclude.
	// Filled in when the key is GOOG_DAYPART_TARGETING.
	DayPartTargetingValue *TargetingValueDayPartTargeting `json:"dayPartTargetingValue,omitempty"`

	// LongValue: The long value to exclude/include.
	LongValue int64 `json:"longValue,omitempty,string"`

	// StringValue: The string value to exclude/include.
	StringValue string `json:"stringValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreativeSizeValue")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TargetingValue) MarshalJSON() ([]byte, error) {
	type noMethod TargetingValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TargetingValueCreativeSize struct {
	// CompanionSizes: For video size type, the list of companion sizes.
	CompanionSizes []*TargetingValueSize `json:"companionSizes,omitempty"`

	// CreativeSizeType: The Creative size type.
	CreativeSizeType string `json:"creativeSizeType,omitempty"`

	// Size: For regular or video creative size type, specifies the size of
	// the creative.
	Size *TargetingValueSize `json:"size,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CompanionSizes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TargetingValueCreativeSize) MarshalJSON() ([]byte, error) {
	type noMethod TargetingValueCreativeSize
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TargetingValueDayPartTargeting struct {
	DayParts []*TargetingValueDayPartTargetingDayPart `json:"dayParts,omitempty"`

	TimeZoneType string `json:"timeZoneType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DayParts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TargetingValueDayPartTargeting) MarshalJSON() ([]byte, error) {
	type noMethod TargetingValueDayPartTargeting
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TargetingValueDayPartTargetingDayPart struct {
	DayOfWeek string `json:"dayOfWeek,omitempty"`

	EndHour int64 `json:"endHour,omitempty"`

	EndMinute int64 `json:"endMinute,omitempty"`

	StartHour int64 `json:"startHour,omitempty"`

	StartMinute int64 `json:"startMinute,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DayOfWeek") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TargetingValueDayPartTargetingDayPart) MarshalJSON() ([]byte, error) {
	type noMethod TargetingValueDayPartTargetingDayPart
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TargetingValueSize struct {
	// Height: The height of the creative.
	Height int64 `json:"height,omitempty"`

	// Width: The width of the creative.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TargetingValueSize) MarshalJSON() ([]byte, error) {
	type noMethod TargetingValueSize
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "adexchangebuyer.accounts.get":

type AccountsGetCall struct {
	s            *Service
	id           int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets one account by ID.
func (r *AccountsService) Get(id int64) *AccountsGetCall {
	c := &AccountsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsGetCall) QuotaUser(quotaUser string) *AccountsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsGetCall) UserIP(userIP string) *AccountsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsGetCall) Fields(s ...googleapi.Field) *AccountsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsGetCall) IfNoneMatch(entityTag string) *AccountsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsGetCall) Context(ctx context.Context) *AccountsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": strconv.FormatInt(c.id, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.accounts.get" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsGetCall) Do() (*Account, error) {
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
	ret := &Account{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves the authenticated user's list of accounts.
func (r *AccountsService) List() *AccountsListCall {
	c := &AccountsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsListCall) QuotaUser(quotaUser string) *AccountsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsListCall) UserIP(userIP string) *AccountsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsListCall) Fields(s ...googleapi.Field) *AccountsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsListCall) IfNoneMatch(entityTag string) *AccountsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsListCall) Context(ctx context.Context) *AccountsListCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.accounts.list" call.
// Exactly one of *AccountsList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountsList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsListCall) Do() (*AccountsList, error) {
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
	ret := &AccountsList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s          *Service
	id         int64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates an existing account. This method supports patch
// semantics.
func (r *AccountsService) Patch(id int64, account *Account) *AccountsPatchCall {
	c := &AccountsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	c.account = account
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsPatchCall) QuotaUser(quotaUser string) *AccountsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsPatchCall) UserIP(userIP string) *AccountsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsPatchCall) Fields(s ...googleapi.Field) *AccountsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsPatchCall) Context(ctx context.Context) *AccountsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": strconv.FormatInt(c.id, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.accounts.patch" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsPatchCall) Do() (*Account, error) {
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
	ret := &Account{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s          *Service
	id         int64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates an existing account.
func (r *AccountsService) Update(id int64, account *Account) *AccountsUpdateCall {
	c := &AccountsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	c.account = account
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsUpdateCall) QuotaUser(quotaUser string) *AccountsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsUpdateCall) UserIP(userIP string) *AccountsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsUpdateCall) Fields(s ...googleapi.Field) *AccountsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsUpdateCall) Context(ctx context.Context) *AccountsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": strconv.FormatInt(c.id, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.accounts.update" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsUpdateCall) Do() (*Account, error) {
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
	ret := &Account{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// method id "adexchangebuyer.billingInfo.get":

type BillingInfoGetCall struct {
	s            *Service
	accountId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns the billing information for one account specified by
// account ID.
func (r *BillingInfoService) Get(accountId int64) *BillingInfoGetCall {
	c := &BillingInfoGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *BillingInfoGetCall) QuotaUser(quotaUser string) *BillingInfoGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *BillingInfoGetCall) UserIP(userIP string) *BillingInfoGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BillingInfoGetCall) Fields(s ...googleapi.Field) *BillingInfoGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BillingInfoGetCall) IfNoneMatch(entityTag string) *BillingInfoGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BillingInfoGetCall) Context(ctx context.Context) *BillingInfoGetCall {
	c.ctx_ = ctx
	return c
}

func (c *BillingInfoGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "billinginfo/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.billingInfo.get" call.
// Exactly one of *BillingInfo or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *BillingInfo.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *BillingInfoGetCall) Do() (*BillingInfo, error) {
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
	ret := &BillingInfo{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the billing information for one account specified by account ID.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.billingInfo.get",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "billinginfo/{accountId}",
	//   "response": {
	//     "$ref": "BillingInfo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.billingInfo.list":

type BillingInfoListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of billing information for all accounts of the
// authenticated user.
func (r *BillingInfoService) List() *BillingInfoListCall {
	c := &BillingInfoListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *BillingInfoListCall) QuotaUser(quotaUser string) *BillingInfoListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *BillingInfoListCall) UserIP(userIP string) *BillingInfoListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BillingInfoListCall) Fields(s ...googleapi.Field) *BillingInfoListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BillingInfoListCall) IfNoneMatch(entityTag string) *BillingInfoListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BillingInfoListCall) Context(ctx context.Context) *BillingInfoListCall {
	c.ctx_ = ctx
	return c
}

func (c *BillingInfoListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "billinginfo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.billingInfo.list" call.
// Exactly one of *BillingInfoList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *BillingInfoList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BillingInfoListCall) Do() (*BillingInfoList, error) {
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
	ret := &BillingInfoList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of billing information for all accounts of the authenticated user.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.billingInfo.list",
	//   "path": "billinginfo",
	//   "response": {
	//     "$ref": "BillingInfoList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.budget.get":

type BudgetGetCall struct {
	s            *Service
	accountId    int64
	billingId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns the budget information for the adgroup specified by the
// accountId and billingId.
func (r *BudgetService) Get(accountId int64, billingId int64) *BudgetGetCall {
	c := &BudgetGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.billingId = billingId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *BudgetGetCall) QuotaUser(quotaUser string) *BudgetGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *BudgetGetCall) UserIP(userIP string) *BudgetGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BudgetGetCall) Fields(s ...googleapi.Field) *BudgetGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BudgetGetCall) IfNoneMatch(entityTag string) *BudgetGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BudgetGetCall) Context(ctx context.Context) *BudgetGetCall {
	c.ctx_ = ctx
	return c
}

func (c *BudgetGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "billinginfo/{accountId}/{billingId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
		"billingId": strconv.FormatInt(c.billingId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.budget.get" call.
// Exactly one of *Budget or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Budget.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BudgetGetCall) Do() (*Budget, error) {
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
	ret := &Budget{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the budget information for the adgroup specified by the accountId and billingId.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.budget.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "billingId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to get the budget information for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "billingId": {
	//       "description": "The billing id to get the budget information for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "billinginfo/{accountId}/{billingId}",
	//   "response": {
	//     "$ref": "Budget"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.budget.patch":

type BudgetPatchCall struct {
	s          *Service
	accountId  int64
	billingId  int64
	budget     *Budget
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates the budget amount for the budget of the adgroup
// specified by the accountId and billingId, with the budget amount in
// the request. This method supports patch semantics.
func (r *BudgetService) Patch(accountId int64, billingId int64, budget *Budget) *BudgetPatchCall {
	c := &BudgetPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.billingId = billingId
	c.budget = budget
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *BudgetPatchCall) QuotaUser(quotaUser string) *BudgetPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *BudgetPatchCall) UserIP(userIP string) *BudgetPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BudgetPatchCall) Fields(s ...googleapi.Field) *BudgetPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BudgetPatchCall) Context(ctx context.Context) *BudgetPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *BudgetPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.budget)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "billinginfo/{accountId}/{billingId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
		"billingId": strconv.FormatInt(c.billingId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.budget.patch" call.
// Exactly one of *Budget or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Budget.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BudgetPatchCall) Do() (*Budget, error) {
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
	ret := &Budget{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the budget amount for the budget of the adgroup specified by the accountId and billingId, with the budget amount in the request. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "adexchangebuyer.budget.patch",
	//   "parameterOrder": [
	//     "accountId",
	//     "billingId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id associated with the budget being updated.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "billingId": {
	//       "description": "The billing id associated with the budget being updated.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "billinginfo/{accountId}/{billingId}",
	//   "request": {
	//     "$ref": "Budget"
	//   },
	//   "response": {
	//     "$ref": "Budget"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.budget.update":

type BudgetUpdateCall struct {
	s          *Service
	accountId  int64
	billingId  int64
	budget     *Budget
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates the budget amount for the budget of the adgroup
// specified by the accountId and billingId, with the budget amount in
// the request.
func (r *BudgetService) Update(accountId int64, billingId int64, budget *Budget) *BudgetUpdateCall {
	c := &BudgetUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.billingId = billingId
	c.budget = budget
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *BudgetUpdateCall) QuotaUser(quotaUser string) *BudgetUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *BudgetUpdateCall) UserIP(userIP string) *BudgetUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BudgetUpdateCall) Fields(s ...googleapi.Field) *BudgetUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BudgetUpdateCall) Context(ctx context.Context) *BudgetUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *BudgetUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.budget)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "billinginfo/{accountId}/{billingId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
		"billingId": strconv.FormatInt(c.billingId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.budget.update" call.
// Exactly one of *Budget or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Budget.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BudgetUpdateCall) Do() (*Budget, error) {
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
	ret := &Budget{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the budget amount for the budget of the adgroup specified by the accountId and billingId, with the budget amount in the request.",
	//   "httpMethod": "PUT",
	//   "id": "adexchangebuyer.budget.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "billingId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id associated with the budget being updated.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "billingId": {
	//       "description": "The billing id associated with the budget being updated.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "billinginfo/{accountId}/{billingId}",
	//   "request": {
	//     "$ref": "Budget"
	//   },
	//   "response": {
	//     "$ref": "Budget"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.creatives.addDeal":

type CreativesAddDealCall struct {
	s               *Service
	accountId       int64
	buyerCreativeId string
	dealId          int64
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// AddDeal: Add a deal id association for the creative.
func (r *CreativesService) AddDeal(accountId int64, buyerCreativeId string, dealId int64) *CreativesAddDealCall {
	c := &CreativesAddDealCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.buyerCreativeId = buyerCreativeId
	c.dealId = dealId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CreativesAddDealCall) QuotaUser(quotaUser string) *CreativesAddDealCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CreativesAddDealCall) UserIP(userIP string) *CreativesAddDealCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CreativesAddDealCall) Fields(s ...googleapi.Field) *CreativesAddDealCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CreativesAddDealCall) Context(ctx context.Context) *CreativesAddDealCall {
	c.ctx_ = ctx
	return c
}

func (c *CreativesAddDealCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives/{accountId}/{buyerCreativeId}/addDeal/{dealId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"buyerCreativeId": c.buyerCreativeId,
		"dealId":          strconv.FormatInt(c.dealId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.creatives.addDeal" call.
func (c *CreativesAddDealCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Add a deal id association for the creative.",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.creatives.addDeal",
	//   "parameterOrder": [
	//     "accountId",
	//     "buyerCreativeId",
	//     "dealId"
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
	//     },
	//     "dealId": {
	//       "description": "The id of the deal id to associate with this creative.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "creatives/{accountId}/{buyerCreativeId}/addDeal/{dealId}",
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
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
}

// Get: Gets the status for a single creative. A creative will be
// available 30-40 minutes after submission.
func (r *CreativesService) Get(accountId int64, buyerCreativeId string) *CreativesGetCall {
	c := &CreativesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.buyerCreativeId = buyerCreativeId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CreativesGetCall) QuotaUser(quotaUser string) *CreativesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CreativesGetCall) UserIP(userIP string) *CreativesGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CreativesGetCall) Fields(s ...googleapi.Field) *CreativesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CreativesGetCall) IfNoneMatch(entityTag string) *CreativesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CreativesGetCall) Context(ctx context.Context) *CreativesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CreativesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives/{accountId}/{buyerCreativeId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"buyerCreativeId": c.buyerCreativeId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.creatives.get" call.
// Exactly one of *Creative or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Creative.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CreativesGetCall) Do() (*Creative, error) {
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
	ret := &Creative{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s          *Service
	creative   *Creative
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Submit a new creative.
func (r *CreativesService) Insert(creative *Creative) *CreativesInsertCall {
	c := &CreativesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.creative = creative
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CreativesInsertCall) QuotaUser(quotaUser string) *CreativesInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CreativesInsertCall) UserIP(userIP string) *CreativesInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CreativesInsertCall) Fields(s ...googleapi.Field) *CreativesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CreativesInsertCall) Context(ctx context.Context) *CreativesInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *CreativesInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.creative)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.creatives.insert" call.
// Exactly one of *Creative or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Creative.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CreativesInsertCall) Do() (*Creative, error) {
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
	ret := &Creative{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of the authenticated user's active creatives.
// A creative will be available 30-40 minutes after submission.
func (r *CreativesService) List() *CreativesListCall {
	c := &CreativesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// AccountId sets the optional parameter "accountId": When specified,
// only creatives for the given account ids are returned.
func (c *CreativesListCall) AccountId(accountId ...int64) *CreativesListCall {
	var accountId_ []string
	for _, v := range accountId {
		accountId_ = append(accountId_, fmt.Sprint(v))
	}
	c.urlParams_.SetMulti("accountId", accountId_)
	return c
}

// BuyerCreativeId sets the optional parameter "buyerCreativeId": When
// specified, only creatives for the given buyer creative ids are
// returned.
func (c *CreativesListCall) BuyerCreativeId(buyerCreativeId ...string) *CreativesListCall {
	c.urlParams_.SetMulti("buyerCreativeId", append([]string{}, buyerCreativeId...))
	return c
}

// DealsStatusFilter sets the optional parameter "dealsStatusFilter":
// When specified, only creatives having the given deals status are
// returned.
//
// Possible values:
//   "approved" - Creatives which have been approved for serving on
// deals.
//   "conditionally_approved" - Creatives which have been conditionally
// approved for serving on deals.
//   "disapproved" - Creatives which have been disapproved for serving
// on deals.
//   "not_checked" - Creatives whose deals status is not yet checked.
func (c *CreativesListCall) DealsStatusFilter(dealsStatusFilter string) *CreativesListCall {
	c.urlParams_.Set("dealsStatusFilter", dealsStatusFilter)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of entries returned on one result page. If not set, the default is
// 100.
func (c *CreativesListCall) MaxResults(maxResults int64) *CreativesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// OpenAuctionStatusFilter sets the optional parameter
// "openAuctionStatusFilter": When specified, only creatives having the
// given open auction status are returned.
//
// Possible values:
//   "approved" - Creatives which have been approved for serving on the
// open auction.
//   "conditionally_approved" - Creatives which have been conditionally
// approved for serving on the open auction.
//   "disapproved" - Creatives which have been disapproved for serving
// on the open auction.
//   "not_checked" - Creatives whose open auction status is not yet
// checked.
func (c *CreativesListCall) OpenAuctionStatusFilter(openAuctionStatusFilter string) *CreativesListCall {
	c.urlParams_.Set("openAuctionStatusFilter", openAuctionStatusFilter)
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad clients. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *CreativesListCall) PageToken(pageToken string) *CreativesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CreativesListCall) QuotaUser(quotaUser string) *CreativesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CreativesListCall) UserIP(userIP string) *CreativesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CreativesListCall) Fields(s ...googleapi.Field) *CreativesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CreativesListCall) IfNoneMatch(entityTag string) *CreativesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CreativesListCall) Context(ctx context.Context) *CreativesListCall {
	c.ctx_ = ctx
	return c
}

func (c *CreativesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.creatives.list" call.
// Exactly one of *CreativesList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CreativesList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CreativesListCall) Do() (*CreativesList, error) {
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
	ret := &CreativesList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of the authenticated user's active creatives. A creative will be available 30-40 minutes after submission.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.creatives.list",
	//   "parameters": {
	//     "accountId": {
	//       "description": "When specified, only creatives for the given account ids are returned.",
	//       "format": "int32",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "integer"
	//     },
	//     "buyerCreativeId": {
	//       "description": "When specified, only creatives for the given buyer creative ids are returned.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "dealsStatusFilter": {
	//       "description": "When specified, only creatives having the given deals status are returned.",
	//       "enum": [
	//         "approved",
	//         "conditionally_approved",
	//         "disapproved",
	//         "not_checked"
	//       ],
	//       "enumDescriptions": [
	//         "Creatives which have been approved for serving on deals.",
	//         "Creatives which have been conditionally approved for serving on deals.",
	//         "Creatives which have been disapproved for serving on deals.",
	//         "Creatives whose deals status is not yet checked."
	//       ],
	//       "location": "query",
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
	//     "openAuctionStatusFilter": {
	//       "description": "When specified, only creatives having the given open auction status are returned.",
	//       "enum": [
	//         "approved",
	//         "conditionally_approved",
	//         "disapproved",
	//         "not_checked"
	//       ],
	//       "enumDescriptions": [
	//         "Creatives which have been approved for serving on the open auction.",
	//         "Creatives which have been conditionally approved for serving on the open auction.",
	//         "Creatives which have been disapproved for serving on the open auction.",
	//         "Creatives whose open auction status is not yet checked."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad clients. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response. Optional.",
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

// method id "adexchangebuyer.creatives.removeDeal":

type CreativesRemoveDealCall struct {
	s               *Service
	accountId       int64
	buyerCreativeId string
	dealId          int64
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// RemoveDeal: Remove a deal id associated with the creative.
func (r *CreativesService) RemoveDeal(accountId int64, buyerCreativeId string, dealId int64) *CreativesRemoveDealCall {
	c := &CreativesRemoveDealCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.buyerCreativeId = buyerCreativeId
	c.dealId = dealId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CreativesRemoveDealCall) QuotaUser(quotaUser string) *CreativesRemoveDealCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CreativesRemoveDealCall) UserIP(userIP string) *CreativesRemoveDealCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CreativesRemoveDealCall) Fields(s ...googleapi.Field) *CreativesRemoveDealCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CreativesRemoveDealCall) Context(ctx context.Context) *CreativesRemoveDealCall {
	c.ctx_ = ctx
	return c
}

func (c *CreativesRemoveDealCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "creatives/{accountId}/{buyerCreativeId}/removeDeal/{dealId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"buyerCreativeId": c.buyerCreativeId,
		"dealId":          strconv.FormatInt(c.dealId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.creatives.removeDeal" call.
func (c *CreativesRemoveDealCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Remove a deal id associated with the creative.",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.creatives.removeDeal",
	//   "parameterOrder": [
	//     "accountId",
	//     "buyerCreativeId",
	//     "dealId"
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
	//     },
	//     "dealId": {
	//       "description": "The id of the deal id to disassociate with this creative.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "creatives/{accountId}/{buyerCreativeId}/removeDeal/{dealId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplacedeals.delete":

type MarketplacedealsDeleteCall struct {
	s                       *Service
	orderId                 string
	deleteorderdealsrequest *DeleteOrderDealsRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
}

// Delete: Delete the specified deals from the order
func (r *MarketplacedealsService) Delete(orderId string, deleteorderdealsrequest *DeleteOrderDealsRequest) *MarketplacedealsDeleteCall {
	c := &MarketplacedealsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	c.deleteorderdealsrequest = deleteorderdealsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplacedealsDeleteCall) QuotaUser(quotaUser string) *MarketplacedealsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplacedealsDeleteCall) UserIP(userIP string) *MarketplacedealsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplacedealsDeleteCall) Fields(s ...googleapi.Field) *MarketplacedealsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplacedealsDeleteCall) Context(ctx context.Context) *MarketplacedealsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplacedealsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.deleteorderdealsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/deals/delete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId": c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplacedeals.delete" call.
// Exactly one of *DeleteOrderDealsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *DeleteOrderDealsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplacedealsDeleteCall) Do() (*DeleteOrderDealsResponse, error) {
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
	ret := &DeleteOrderDealsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Delete the specified deals from the order",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.marketplacedeals.delete",
	//   "parameterOrder": [
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "The orderId to delete deals from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/deals/delete",
	//   "request": {
	//     "$ref": "DeleteOrderDealsRequest"
	//   },
	//   "response": {
	//     "$ref": "DeleteOrderDealsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplacedeals.insert":

type MarketplacedealsInsertCall struct {
	s                    *Service
	orderId              string
	addorderdealsrequest *AddOrderDealsRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
}

// Insert: Add new deals for the specified order
func (r *MarketplacedealsService) Insert(orderId string, addorderdealsrequest *AddOrderDealsRequest) *MarketplacedealsInsertCall {
	c := &MarketplacedealsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	c.addorderdealsrequest = addorderdealsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplacedealsInsertCall) QuotaUser(quotaUser string) *MarketplacedealsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplacedealsInsertCall) UserIP(userIP string) *MarketplacedealsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplacedealsInsertCall) Fields(s ...googleapi.Field) *MarketplacedealsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplacedealsInsertCall) Context(ctx context.Context) *MarketplacedealsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplacedealsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.addorderdealsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/deals/insert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId": c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplacedeals.insert" call.
// Exactly one of *AddOrderDealsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AddOrderDealsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplacedealsInsertCall) Do() (*AddOrderDealsResponse, error) {
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
	ret := &AddOrderDealsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add new deals for the specified order",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.marketplacedeals.insert",
	//   "parameterOrder": [
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "OrderId for which deals need to be added.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/deals/insert",
	//   "request": {
	//     "$ref": "AddOrderDealsRequest"
	//   },
	//   "response": {
	//     "$ref": "AddOrderDealsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplacedeals.list":

type MarketplacedealsListCall struct {
	s            *Service
	orderId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: List all the deals for a given order
func (r *MarketplacedealsService) List(orderId string) *MarketplacedealsListCall {
	c := &MarketplacedealsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplacedealsListCall) QuotaUser(quotaUser string) *MarketplacedealsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplacedealsListCall) UserIP(userIP string) *MarketplacedealsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplacedealsListCall) Fields(s ...googleapi.Field) *MarketplacedealsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MarketplacedealsListCall) IfNoneMatch(entityTag string) *MarketplacedealsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplacedealsListCall) Context(ctx context.Context) *MarketplacedealsListCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplacedealsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/deals")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId": c.orderId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplacedeals.list" call.
// Exactly one of *GetOrderDealsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetOrderDealsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplacedealsListCall) Do() (*GetOrderDealsResponse, error) {
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
	ret := &GetOrderDealsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all the deals for a given order",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.marketplacedeals.list",
	//   "parameterOrder": [
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "The orderId to get deals for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/deals",
	//   "response": {
	//     "$ref": "GetOrderDealsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplacedeals.update":

type MarketplacedealsUpdateCall struct {
	s                        *Service
	orderId                  string
	editallorderdealsrequest *EditAllOrderDealsRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
}

// Update: Replaces all the deals in the order with the passed in deals
func (r *MarketplacedealsService) Update(orderId string, editallorderdealsrequest *EditAllOrderDealsRequest) *MarketplacedealsUpdateCall {
	c := &MarketplacedealsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	c.editallorderdealsrequest = editallorderdealsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplacedealsUpdateCall) QuotaUser(quotaUser string) *MarketplacedealsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplacedealsUpdateCall) UserIP(userIP string) *MarketplacedealsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplacedealsUpdateCall) Fields(s ...googleapi.Field) *MarketplacedealsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplacedealsUpdateCall) Context(ctx context.Context) *MarketplacedealsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplacedealsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.editallorderdealsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/deals/update")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId": c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplacedeals.update" call.
// Exactly one of *EditAllOrderDealsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *EditAllOrderDealsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplacedealsUpdateCall) Do() (*EditAllOrderDealsResponse, error) {
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
	ret := &EditAllOrderDealsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Replaces all the deals in the order with the passed in deals",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.marketplacedeals.update",
	//   "parameterOrder": [
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "The orderId to edit deals on.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/deals/update",
	//   "request": {
	//     "$ref": "EditAllOrderDealsRequest"
	//   },
	//   "response": {
	//     "$ref": "EditAllOrderDealsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplacenotes.insert":

type MarketplacenotesInsertCall struct {
	s                    *Service
	orderId              string
	addordernotesrequest *AddOrderNotesRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
}

// Insert: Add notes to the order
func (r *MarketplacenotesService) Insert(orderId string, addordernotesrequest *AddOrderNotesRequest) *MarketplacenotesInsertCall {
	c := &MarketplacenotesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	c.addordernotesrequest = addordernotesrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplacenotesInsertCall) QuotaUser(quotaUser string) *MarketplacenotesInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplacenotesInsertCall) UserIP(userIP string) *MarketplacenotesInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplacenotesInsertCall) Fields(s ...googleapi.Field) *MarketplacenotesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplacenotesInsertCall) Context(ctx context.Context) *MarketplacenotesInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplacenotesInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.addordernotesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/notes/insert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId": c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplacenotes.insert" call.
// Exactly one of *AddOrderNotesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AddOrderNotesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplacenotesInsertCall) Do() (*AddOrderNotesResponse, error) {
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
	ret := &AddOrderNotesResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add notes to the order",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.marketplacenotes.insert",
	//   "parameterOrder": [
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "The orderId to add notes for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/notes/insert",
	//   "request": {
	//     "$ref": "AddOrderNotesRequest"
	//   },
	//   "response": {
	//     "$ref": "AddOrderNotesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplacenotes.list":

type MarketplacenotesListCall struct {
	s            *Service
	orderId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Get all the notes associated with an order
func (r *MarketplacenotesService) List(orderId string) *MarketplacenotesListCall {
	c := &MarketplacenotesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplacenotesListCall) QuotaUser(quotaUser string) *MarketplacenotesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplacenotesListCall) UserIP(userIP string) *MarketplacenotesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplacenotesListCall) Fields(s ...googleapi.Field) *MarketplacenotesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MarketplacenotesListCall) IfNoneMatch(entityTag string) *MarketplacenotesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplacenotesListCall) Context(ctx context.Context) *MarketplacenotesListCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplacenotesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/notes")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId": c.orderId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplacenotes.list" call.
// Exactly one of *GetOrderNotesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetOrderNotesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplacenotesListCall) Do() (*GetOrderNotesResponse, error) {
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
	ret := &GetOrderNotesResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get all the notes associated with an order",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.marketplacenotes.list",
	//   "parameterOrder": [
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "The orderId to get notes for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/notes",
	//   "response": {
	//     "$ref": "GetOrderNotesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplaceoffers.get":

type MarketplaceoffersGetCall struct {
	s            *Service
	offerId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the requested negotiation.
func (r *MarketplaceoffersService) Get(offerId string) *MarketplaceoffersGetCall {
	c := &MarketplaceoffersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.offerId = offerId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplaceoffersGetCall) QuotaUser(quotaUser string) *MarketplaceoffersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplaceoffersGetCall) UserIP(userIP string) *MarketplaceoffersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplaceoffersGetCall) Fields(s ...googleapi.Field) *MarketplaceoffersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MarketplaceoffersGetCall) IfNoneMatch(entityTag string) *MarketplaceoffersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplaceoffersGetCall) Context(ctx context.Context) *MarketplaceoffersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplaceoffersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOffers/{offerId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"offerId": c.offerId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplaceoffers.get" call.
// Exactly one of *MarketplaceOffer or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *MarketplaceOffer.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplaceoffersGetCall) Do() (*MarketplaceOffer, error) {
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
	ret := &MarketplaceOffer{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the requested negotiation.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.marketplaceoffers.get",
	//   "parameterOrder": [
	//     "offerId"
	//   ],
	//   "parameters": {
	//     "offerId": {
	//       "description": "The offerId for the offer to get the head revision for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOffers/{offerId}",
	//   "response": {
	//     "$ref": "MarketplaceOffer"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplaceoffers.search":

type MarketplaceoffersSearchCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Search: Gets the requested negotiation.
func (r *MarketplaceoffersService) Search() *MarketplaceoffersSearchCall {
	c := &MarketplaceoffersSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PqlQuery sets the optional parameter "pqlQuery": The pql query used
// to query for offers.
func (c *MarketplaceoffersSearchCall) PqlQuery(pqlQuery string) *MarketplaceoffersSearchCall {
	c.urlParams_.Set("pqlQuery", pqlQuery)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplaceoffersSearchCall) QuotaUser(quotaUser string) *MarketplaceoffersSearchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplaceoffersSearchCall) UserIP(userIP string) *MarketplaceoffersSearchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplaceoffersSearchCall) Fields(s ...googleapi.Field) *MarketplaceoffersSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MarketplaceoffersSearchCall) IfNoneMatch(entityTag string) *MarketplaceoffersSearchCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplaceoffersSearchCall) Context(ctx context.Context) *MarketplaceoffersSearchCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplaceoffersSearchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOffers/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplaceoffers.search" call.
// Exactly one of *GetOffersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetOffersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplaceoffersSearchCall) Do() (*GetOffersResponse, error) {
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
	ret := &GetOffersResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the requested negotiation.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.marketplaceoffers.search",
	//   "parameters": {
	//     "pqlQuery": {
	//       "description": "The pql query used to query for offers.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOffers/search",
	//   "response": {
	//     "$ref": "GetOffersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplaceorders.get":

type MarketplaceordersGetCall struct {
	s            *Service
	orderId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Get an order given its id
func (r *MarketplaceordersService) Get(orderId string) *MarketplaceordersGetCall {
	c := &MarketplaceordersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplaceordersGetCall) QuotaUser(quotaUser string) *MarketplaceordersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplaceordersGetCall) UserIP(userIP string) *MarketplaceordersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplaceordersGetCall) Fields(s ...googleapi.Field) *MarketplaceordersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MarketplaceordersGetCall) IfNoneMatch(entityTag string) *MarketplaceordersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplaceordersGetCall) Context(ctx context.Context) *MarketplaceordersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplaceordersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId": c.orderId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplaceorders.get" call.
// Exactly one of *MarketplaceOrder or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *MarketplaceOrder.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplaceordersGetCall) Do() (*MarketplaceOrder, error) {
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
	ret := &MarketplaceOrder{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get an order given its id",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.marketplaceorders.get",
	//   "parameterOrder": [
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "Id of the order to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}",
	//   "response": {
	//     "$ref": "MarketplaceOrder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplaceorders.insert":

type MarketplaceordersInsertCall struct {
	s                   *Service
	createordersrequest *CreateOrdersRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// Insert: Create the given list of orders
func (r *MarketplaceordersService) Insert(createordersrequest *CreateOrdersRequest) *MarketplaceordersInsertCall {
	c := &MarketplaceordersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.createordersrequest = createordersrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplaceordersInsertCall) QuotaUser(quotaUser string) *MarketplaceordersInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplaceordersInsertCall) UserIP(userIP string) *MarketplaceordersInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplaceordersInsertCall) Fields(s ...googleapi.Field) *MarketplaceordersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplaceordersInsertCall) Context(ctx context.Context) *MarketplaceordersInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplaceordersInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createordersrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/insert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplaceorders.insert" call.
// Exactly one of *CreateOrdersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *CreateOrdersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplaceordersInsertCall) Do() (*CreateOrdersResponse, error) {
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
	ret := &CreateOrdersResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create the given list of orders",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.marketplaceorders.insert",
	//   "path": "marketplaceOrders/insert",
	//   "request": {
	//     "$ref": "CreateOrdersRequest"
	//   },
	//   "response": {
	//     "$ref": "CreateOrdersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplaceorders.patch":

type MarketplaceordersPatchCall struct {
	s                *Service
	orderId          string
	revisionNumber   int64
	updateAction     string
	marketplaceorder *MarketplaceOrder
	urlParams_       gensupport.URLParams
	ctx_             context.Context
}

// Patch: Update the given order. This method supports patch semantics.
func (r *MarketplaceordersService) Patch(orderId string, revisionNumber int64, updateAction string, marketplaceorder *MarketplaceOrder) *MarketplaceordersPatchCall {
	c := &MarketplaceordersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	c.revisionNumber = revisionNumber
	c.updateAction = updateAction
	c.marketplaceorder = marketplaceorder
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplaceordersPatchCall) QuotaUser(quotaUser string) *MarketplaceordersPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplaceordersPatchCall) UserIP(userIP string) *MarketplaceordersPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplaceordersPatchCall) Fields(s ...googleapi.Field) *MarketplaceordersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplaceordersPatchCall) Context(ctx context.Context) *MarketplaceordersPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplaceordersPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.marketplaceorder)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/{revisionNumber}/{updateAction}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId":        c.orderId,
		"revisionNumber": strconv.FormatInt(c.revisionNumber, 10),
		"updateAction":   c.updateAction,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplaceorders.patch" call.
// Exactly one of *MarketplaceOrder or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *MarketplaceOrder.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplaceordersPatchCall) Do() (*MarketplaceOrder, error) {
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
	ret := &MarketplaceOrder{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update the given order. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "adexchangebuyer.marketplaceorders.patch",
	//   "parameterOrder": [
	//     "orderId",
	//     "revisionNumber",
	//     "updateAction"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "The order id to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "revisionNumber": {
	//       "description": "The last known revision number to update. If the head revision in the marketplace database has since changed, an error will be thrown. The caller should then fetch the lastest order at head revision and retry the update at that revision.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateAction": {
	//       "description": "The proposed action to take on the order.",
	//       "enum": [
	//         "accept",
	//         "cancel",
	//         "propose",
	//         "unknownAction",
	//         "updateFinalized"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/{revisionNumber}/{updateAction}",
	//   "request": {
	//     "$ref": "MarketplaceOrder"
	//   },
	//   "response": {
	//     "$ref": "MarketplaceOrder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplaceorders.search":

type MarketplaceordersSearchCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Search: Search for orders using pql query
func (r *MarketplaceordersService) Search() *MarketplaceordersSearchCall {
	c := &MarketplaceordersSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PqlQuery sets the optional parameter "pqlQuery": Query string to
// retrieve specific orders.
func (c *MarketplaceordersSearchCall) PqlQuery(pqlQuery string) *MarketplaceordersSearchCall {
	c.urlParams_.Set("pqlQuery", pqlQuery)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplaceordersSearchCall) QuotaUser(quotaUser string) *MarketplaceordersSearchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplaceordersSearchCall) UserIP(userIP string) *MarketplaceordersSearchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplaceordersSearchCall) Fields(s ...googleapi.Field) *MarketplaceordersSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MarketplaceordersSearchCall) IfNoneMatch(entityTag string) *MarketplaceordersSearchCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplaceordersSearchCall) Context(ctx context.Context) *MarketplaceordersSearchCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplaceordersSearchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplaceorders.search" call.
// Exactly one of *GetOrdersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetOrdersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplaceordersSearchCall) Do() (*GetOrdersResponse, error) {
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
	ret := &GetOrdersResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Search for orders using pql query",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.marketplaceorders.search",
	//   "parameters": {
	//     "pqlQuery": {
	//       "description": "Query string to retrieve specific orders.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/search",
	//   "response": {
	//     "$ref": "GetOrdersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.marketplaceorders.update":

type MarketplaceordersUpdateCall struct {
	s                *Service
	orderId          string
	revisionNumber   int64
	updateAction     string
	marketplaceorder *MarketplaceOrder
	urlParams_       gensupport.URLParams
	ctx_             context.Context
}

// Update: Update the given order
func (r *MarketplaceordersService) Update(orderId string, revisionNumber int64, updateAction string, marketplaceorder *MarketplaceOrder) *MarketplaceordersUpdateCall {
	c := &MarketplaceordersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderId = orderId
	c.revisionNumber = revisionNumber
	c.updateAction = updateAction
	c.marketplaceorder = marketplaceorder
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *MarketplaceordersUpdateCall) QuotaUser(quotaUser string) *MarketplaceordersUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *MarketplaceordersUpdateCall) UserIP(userIP string) *MarketplaceordersUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MarketplaceordersUpdateCall) Fields(s ...googleapi.Field) *MarketplaceordersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MarketplaceordersUpdateCall) Context(ctx context.Context) *MarketplaceordersUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *MarketplaceordersUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.marketplaceorder)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "marketplaceOrders/{orderId}/{revisionNumber}/{updateAction}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"orderId":        c.orderId,
		"revisionNumber": strconv.FormatInt(c.revisionNumber, 10),
		"updateAction":   c.updateAction,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.marketplaceorders.update" call.
// Exactly one of *MarketplaceOrder or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *MarketplaceOrder.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MarketplaceordersUpdateCall) Do() (*MarketplaceOrder, error) {
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
	ret := &MarketplaceOrder{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update the given order",
	//   "httpMethod": "PUT",
	//   "id": "adexchangebuyer.marketplaceorders.update",
	//   "parameterOrder": [
	//     "orderId",
	//     "revisionNumber",
	//     "updateAction"
	//   ],
	//   "parameters": {
	//     "orderId": {
	//       "description": "The order id to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "revisionNumber": {
	//       "description": "The last known revision number to update. If the head revision in the marketplace database has since changed, an error will be thrown. The caller should then fetch the lastest order at head revision and retry the update at that revision.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateAction": {
	//       "description": "The proposed action to take on the order.",
	//       "enum": [
	//         "accept",
	//         "cancel",
	//         "propose",
	//         "unknownAction",
	//         "updateFinalized"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "marketplaceOrders/{orderId}/{revisionNumber}/{updateAction}",
	//   "request": {
	//     "$ref": "MarketplaceOrder"
	//   },
	//   "response": {
	//     "$ref": "MarketplaceOrder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.performanceReport.list":

type PerformanceReportListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves the authenticated user's list of performance metrics.
func (r *PerformanceReportService) List(accountId int64, endDateTime string, startDateTime string) *PerformanceReportListCall {
	c := &PerformanceReportListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("accountId", fmt.Sprint(accountId))
	c.urlParams_.Set("endDateTime", endDateTime)
	c.urlParams_.Set("startDateTime", startDateTime)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of entries returned on one result page. If not set, the default is
// 100.
func (c *PerformanceReportListCall) MaxResults(maxResults int64) *PerformanceReportListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through performance reports. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *PerformanceReportListCall) PageToken(pageToken string) *PerformanceReportListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PerformanceReportListCall) QuotaUser(quotaUser string) *PerformanceReportListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PerformanceReportListCall) UserIP(userIP string) *PerformanceReportListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PerformanceReportListCall) Fields(s ...googleapi.Field) *PerformanceReportListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PerformanceReportListCall) IfNoneMatch(entityTag string) *PerformanceReportListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PerformanceReportListCall) Context(ctx context.Context) *PerformanceReportListCall {
	c.ctx_ = ctx
	return c
}

func (c *PerformanceReportListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "performancereport")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.performanceReport.list" call.
// Exactly one of *PerformanceReportList or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PerformanceReportList.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PerformanceReportListCall) Do() (*PerformanceReportList, error) {
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
	ret := &PerformanceReportList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// method id "adexchangebuyer.pretargetingConfig.delete":

type PretargetingConfigDeleteCall struct {
	s          *Service
	accountId  int64
	configId   int64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes an existing pretargeting config.
func (r *PretargetingConfigService) Delete(accountId int64, configId int64) *PretargetingConfigDeleteCall {
	c := &PretargetingConfigDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.configId = configId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PretargetingConfigDeleteCall) QuotaUser(quotaUser string) *PretargetingConfigDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PretargetingConfigDeleteCall) UserIP(userIP string) *PretargetingConfigDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PretargetingConfigDeleteCall) Fields(s ...googleapi.Field) *PretargetingConfigDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PretargetingConfigDeleteCall) Context(ctx context.Context) *PretargetingConfigDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *PretargetingConfigDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "pretargetingconfigs/{accountId}/{configId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
		"configId":  strconv.FormatInt(c.configId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.pretargetingConfig.delete" call.
func (c *PretargetingConfigDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes an existing pretargeting config.",
	//   "httpMethod": "DELETE",
	//   "id": "adexchangebuyer.pretargetingConfig.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "configId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to delete the pretargeting config for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "configId": {
	//       "description": "The specific id of the configuration to delete.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "pretargetingconfigs/{accountId}/{configId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.pretargetingConfig.get":

type PretargetingConfigGetCall struct {
	s            *Service
	accountId    int64
	configId     int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets a specific pretargeting configuration
func (r *PretargetingConfigService) Get(accountId int64, configId int64) *PretargetingConfigGetCall {
	c := &PretargetingConfigGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.configId = configId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PretargetingConfigGetCall) QuotaUser(quotaUser string) *PretargetingConfigGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PretargetingConfigGetCall) UserIP(userIP string) *PretargetingConfigGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PretargetingConfigGetCall) Fields(s ...googleapi.Field) *PretargetingConfigGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PretargetingConfigGetCall) IfNoneMatch(entityTag string) *PretargetingConfigGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PretargetingConfigGetCall) Context(ctx context.Context) *PretargetingConfigGetCall {
	c.ctx_ = ctx
	return c
}

func (c *PretargetingConfigGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "pretargetingconfigs/{accountId}/{configId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
		"configId":  strconv.FormatInt(c.configId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.pretargetingConfig.get" call.
// Exactly one of *PretargetingConfig or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PretargetingConfig.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PretargetingConfigGetCall) Do() (*PretargetingConfig, error) {
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
	ret := &PretargetingConfig{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a specific pretargeting configuration",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.pretargetingConfig.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "configId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to get the pretargeting config for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "configId": {
	//       "description": "The specific id of the configuration to retrieve.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "pretargetingconfigs/{accountId}/{configId}",
	//   "response": {
	//     "$ref": "PretargetingConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.pretargetingConfig.insert":

type PretargetingConfigInsertCall struct {
	s                  *Service
	accountId          int64
	pretargetingconfig *PretargetingConfig
	urlParams_         gensupport.URLParams
	ctx_               context.Context
}

// Insert: Inserts a new pretargeting configuration.
func (r *PretargetingConfigService) Insert(accountId int64, pretargetingconfig *PretargetingConfig) *PretargetingConfigInsertCall {
	c := &PretargetingConfigInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.pretargetingconfig = pretargetingconfig
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PretargetingConfigInsertCall) QuotaUser(quotaUser string) *PretargetingConfigInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PretargetingConfigInsertCall) UserIP(userIP string) *PretargetingConfigInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PretargetingConfigInsertCall) Fields(s ...googleapi.Field) *PretargetingConfigInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PretargetingConfigInsertCall) Context(ctx context.Context) *PretargetingConfigInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *PretargetingConfigInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pretargetingconfig)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "pretargetingconfigs/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.pretargetingConfig.insert" call.
// Exactly one of *PretargetingConfig or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PretargetingConfig.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PretargetingConfigInsertCall) Do() (*PretargetingConfig, error) {
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
	ret := &PretargetingConfig{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new pretargeting configuration.",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer.pretargetingConfig.insert",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to insert the pretargeting config for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "pretargetingconfigs/{accountId}",
	//   "request": {
	//     "$ref": "PretargetingConfig"
	//   },
	//   "response": {
	//     "$ref": "PretargetingConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.pretargetingConfig.list":

type PretargetingConfigListCall struct {
	s            *Service
	accountId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of the authenticated user's pretargeting
// configurations.
func (r *PretargetingConfigService) List(accountId int64) *PretargetingConfigListCall {
	c := &PretargetingConfigListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PretargetingConfigListCall) QuotaUser(quotaUser string) *PretargetingConfigListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PretargetingConfigListCall) UserIP(userIP string) *PretargetingConfigListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PretargetingConfigListCall) Fields(s ...googleapi.Field) *PretargetingConfigListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PretargetingConfigListCall) IfNoneMatch(entityTag string) *PretargetingConfigListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PretargetingConfigListCall) Context(ctx context.Context) *PretargetingConfigListCall {
	c.ctx_ = ctx
	return c
}

func (c *PretargetingConfigListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "pretargetingconfigs/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.pretargetingConfig.list" call.
// Exactly one of *PretargetingConfigList or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PretargetingConfigList.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PretargetingConfigListCall) Do() (*PretargetingConfigList, error) {
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
	ret := &PretargetingConfigList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of the authenticated user's pretargeting configurations.",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer.pretargetingConfig.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to get the pretargeting configs for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "pretargetingconfigs/{accountId}",
	//   "response": {
	//     "$ref": "PretargetingConfigList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.pretargetingConfig.patch":

type PretargetingConfigPatchCall struct {
	s                  *Service
	accountId          int64
	configId           int64
	pretargetingconfig *PretargetingConfig
	urlParams_         gensupport.URLParams
	ctx_               context.Context
}

// Patch: Updates an existing pretargeting config. This method supports
// patch semantics.
func (r *PretargetingConfigService) Patch(accountId int64, configId int64, pretargetingconfig *PretargetingConfig) *PretargetingConfigPatchCall {
	c := &PretargetingConfigPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.configId = configId
	c.pretargetingconfig = pretargetingconfig
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PretargetingConfigPatchCall) QuotaUser(quotaUser string) *PretargetingConfigPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PretargetingConfigPatchCall) UserIP(userIP string) *PretargetingConfigPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PretargetingConfigPatchCall) Fields(s ...googleapi.Field) *PretargetingConfigPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PretargetingConfigPatchCall) Context(ctx context.Context) *PretargetingConfigPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *PretargetingConfigPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pretargetingconfig)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "pretargetingconfigs/{accountId}/{configId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
		"configId":  strconv.FormatInt(c.configId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.pretargetingConfig.patch" call.
// Exactly one of *PretargetingConfig or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PretargetingConfig.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PretargetingConfigPatchCall) Do() (*PretargetingConfig, error) {
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
	ret := &PretargetingConfig{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing pretargeting config. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "adexchangebuyer.pretargetingConfig.patch",
	//   "parameterOrder": [
	//     "accountId",
	//     "configId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to update the pretargeting config for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "configId": {
	//       "description": "The specific id of the configuration to update.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "pretargetingconfigs/{accountId}/{configId}",
	//   "request": {
	//     "$ref": "PretargetingConfig"
	//   },
	//   "response": {
	//     "$ref": "PretargetingConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer.pretargetingConfig.update":

type PretargetingConfigUpdateCall struct {
	s                  *Service
	accountId          int64
	configId           int64
	pretargetingconfig *PretargetingConfig
	urlParams_         gensupport.URLParams
	ctx_               context.Context
}

// Update: Updates an existing pretargeting config.
func (r *PretargetingConfigService) Update(accountId int64, configId int64, pretargetingconfig *PretargetingConfig) *PretargetingConfigUpdateCall {
	c := &PretargetingConfigUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.configId = configId
	c.pretargetingconfig = pretargetingconfig
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PretargetingConfigUpdateCall) QuotaUser(quotaUser string) *PretargetingConfigUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PretargetingConfigUpdateCall) UserIP(userIP string) *PretargetingConfigUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PretargetingConfigUpdateCall) Fields(s ...googleapi.Field) *PretargetingConfigUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PretargetingConfigUpdateCall) Context(ctx context.Context) *PretargetingConfigUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *PretargetingConfigUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pretargetingconfig)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "pretargetingconfigs/{accountId}/{configId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
		"configId":  strconv.FormatInt(c.configId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "adexchangebuyer.pretargetingConfig.update" call.
// Exactly one of *PretargetingConfig or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PretargetingConfig.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PretargetingConfigUpdateCall) Do() (*PretargetingConfig, error) {
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
	ret := &PretargetingConfig{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing pretargeting config.",
	//   "httpMethod": "PUT",
	//   "id": "adexchangebuyer.pretargetingConfig.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "configId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account id to update the pretargeting config for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "configId": {
	//       "description": "The specific id of the configuration to update.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "pretargetingconfigs/{accountId}/{configId}",
	//   "request": {
	//     "$ref": "PretargetingConfig"
	//   },
	//   "response": {
	//     "$ref": "PretargetingConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}
