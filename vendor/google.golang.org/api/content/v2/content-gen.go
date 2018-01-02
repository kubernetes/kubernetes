// Package content provides access to the Content API for Shopping.
//
// See https://developers.google.com/shopping-content
//
// Usage example:
//
//   import "google.golang.org/api/content/v2"
//   ...
//   contentService, err := content.New(oauthHttpClient)
package content // import "google.golang.org/api/content/v2"

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

const apiId = "content:v2"
const apiName = "content"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/content/v2/"

// OAuth2 scopes used by this API.
const (
	// Manage your product listings and accounts for Google Shopping
	ContentScope = "https://www.googleapis.com/auth/content"
)

func New(client *http.Client) (*APIService, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &APIService{client: client, BasePath: basePath}
	s.Accounts = NewAccountsService(s)
	s.Accountstatuses = NewAccountstatusesService(s)
	s.Accounttax = NewAccounttaxService(s)
	s.Datafeeds = NewDatafeedsService(s)
	s.Datafeedstatuses = NewDatafeedstatusesService(s)
	s.Inventory = NewInventoryService(s)
	s.Orders = NewOrdersService(s)
	s.Products = NewProductsService(s)
	s.Productstatuses = NewProductstatusesService(s)
	s.Shippingsettings = NewShippingsettingsService(s)
	return s, nil
}

type APIService struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Accounts *AccountsService

	Accountstatuses *AccountstatusesService

	Accounttax *AccounttaxService

	Datafeeds *DatafeedsService

	Datafeedstatuses *DatafeedstatusesService

	Inventory *InventoryService

	Orders *OrdersService

	Products *ProductsService

	Productstatuses *ProductstatusesService

	Shippingsettings *ShippingsettingsService
}

func (s *APIService) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAccountsService(s *APIService) *AccountsService {
	rs := &AccountsService{s: s}
	return rs
}

type AccountsService struct {
	s *APIService
}

func NewAccountstatusesService(s *APIService) *AccountstatusesService {
	rs := &AccountstatusesService{s: s}
	return rs
}

type AccountstatusesService struct {
	s *APIService
}

func NewAccounttaxService(s *APIService) *AccounttaxService {
	rs := &AccounttaxService{s: s}
	return rs
}

type AccounttaxService struct {
	s *APIService
}

func NewDatafeedsService(s *APIService) *DatafeedsService {
	rs := &DatafeedsService{s: s}
	return rs
}

type DatafeedsService struct {
	s *APIService
}

func NewDatafeedstatusesService(s *APIService) *DatafeedstatusesService {
	rs := &DatafeedstatusesService{s: s}
	return rs
}

type DatafeedstatusesService struct {
	s *APIService
}

func NewInventoryService(s *APIService) *InventoryService {
	rs := &InventoryService{s: s}
	return rs
}

type InventoryService struct {
	s *APIService
}

func NewOrdersService(s *APIService) *OrdersService {
	rs := &OrdersService{s: s}
	return rs
}

type OrdersService struct {
	s *APIService
}

func NewProductsService(s *APIService) *ProductsService {
	rs := &ProductsService{s: s}
	return rs
}

type ProductsService struct {
	s *APIService
}

func NewProductstatusesService(s *APIService) *ProductstatusesService {
	rs := &ProductstatusesService{s: s}
	return rs
}

type ProductstatusesService struct {
	s *APIService
}

func NewShippingsettingsService(s *APIService) *ShippingsettingsService {
	rs := &ShippingsettingsService{s: s}
	return rs
}

type ShippingsettingsService struct {
	s *APIService
}

// Account: Account data.
type Account struct {
	// AdultContent: Indicates whether the merchant sells adult content.
	AdultContent bool `json:"adultContent,omitempty"`

	// AdwordsLinks: List of linked AdWords accounts that are active or
	// pending approval. To create a new link request, add a new link with
	// status active to the list. It will remain in a pending state until
	// approved or rejected either in the AdWords interface or through the
	// AdWords API. To delete an active link, or to cancel a link request,
	// remove it from the list.
	AdwordsLinks []*AccountAdwordsLink `json:"adwordsLinks,omitempty"`

	// Id: Merchant Center account ID.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#account".
	Kind string `json:"kind,omitempty"`

	// Name: Display name for the account.
	Name string `json:"name,omitempty"`

	// ReviewsUrl: URL for individual seller reviews, i.e., reviews for each
	// child account.
	ReviewsUrl string `json:"reviewsUrl,omitempty"`

	// SellerId: Client-specific, locally-unique, internal ID for the child
	// account.
	SellerId string `json:"sellerId,omitempty"`

	// Users: Users with access to the account. Every account (except for
	// subaccounts) must have at least one admin user.
	Users []*AccountUser `json:"users,omitempty"`

	// WebsiteUrl: The merchant's website.
	WebsiteUrl string `json:"websiteUrl,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AdultContent") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdultContent") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Account) MarshalJSON() ([]byte, error) {
	type noMethod Account
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountAdwordsLink struct {
	// AdwordsId: Customer ID of the AdWords account.
	AdwordsId uint64 `json:"adwordsId,omitempty,string"`

	// Status: Status of the link between this Merchant Center account and
	// the AdWords account. Upon retrieval, it represents the actual status
	// of the link and can be either active if it was approved in Google
	// AdWords or pending if it's pending approval. Upon insertion, it
	// represents the intended status of the link. Re-uploading a link with
	// status active when it's still pending or with status pending when
	// it's already active will have no effect: the status will remain
	// unchanged. Re-uploading a link with deprecated status inactive is
	// equivalent to not submitting the link at all and will delete the link
	// if it was active or cancel the link request if it was pending.
	Status string `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdwordsId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdwordsId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountAdwordsLink) MarshalJSON() ([]byte, error) {
	type noMethod AccountAdwordsLink
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountIdentifier struct {
	// AggregatorId: The aggregator ID, set for aggregators and subaccounts
	// (in that case, it represents the aggregator of the subaccount).
	AggregatorId uint64 `json:"aggregatorId,omitempty,string"`

	// MerchantId: The merchant account ID, set for individual accounts and
	// subaccounts.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "AggregatorId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AggregatorId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountIdentifier) MarshalJSON() ([]byte, error) {
	type noMethod AccountIdentifier
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountStatus: The status of an account, i.e., information about its
// products, which is computed offline and not returned immediately at
// insertion time.
type AccountStatus struct {
	// AccountId: The ID of the account for which the status is reported.
	AccountId string `json:"accountId,omitempty"`

	// AccountLevelIssues: A list of account level issues.
	AccountLevelIssues []*AccountStatusAccountLevelIssue `json:"accountLevelIssues,omitempty"`

	// DataQualityIssues: A list of data quality issues.
	DataQualityIssues []*AccountStatusDataQualityIssue `json:"dataQualityIssues,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountStatus".
	Kind string `json:"kind,omitempty"`

	// WebsiteClaimed: Whether the account's website is claimed or not.
	WebsiteClaimed bool `json:"websiteClaimed,omitempty"`

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

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountStatus) MarshalJSON() ([]byte, error) {
	type noMethod AccountStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountStatusAccountLevelIssue struct {
	// Country: Country for which this issue is reported.
	Country string `json:"country,omitempty"`

	// Detail: Additional details about the issue.
	Detail string `json:"detail,omitempty"`

	// Id: Issue identifier.
	Id string `json:"id,omitempty"`

	// Severity: Severity of the issue.
	Severity string `json:"severity,omitempty"`

	// Title: Short description of the issue.
	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountStatusAccountLevelIssue) MarshalJSON() ([]byte, error) {
	type noMethod AccountStatusAccountLevelIssue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountStatusDataQualityIssue struct {
	// Country: Country for which this issue is reported.
	Country string `json:"country,omitempty"`

	// Detail: A more detailed description of the issue.
	Detail string `json:"detail,omitempty"`

	// DisplayedValue: Actual value displayed on the landing page.
	DisplayedValue string `json:"displayedValue,omitempty"`

	// ExampleItems: Example items featuring the issue.
	ExampleItems []*AccountStatusExampleItem `json:"exampleItems,omitempty"`

	// Id: Issue identifier.
	Id string `json:"id,omitempty"`

	// LastChecked: Last time the account was checked for this issue.
	LastChecked string `json:"lastChecked,omitempty"`

	// Location: The attribute name that is relevant for the issue.
	Location string `json:"location,omitempty"`

	// NumItems: Number of items in the account found to have the said
	// issue.
	NumItems int64 `json:"numItems,omitempty"`

	// Severity: Severity of the problem.
	Severity string `json:"severity,omitempty"`

	// SubmittedValue: Submitted value that causes the issue.
	SubmittedValue string `json:"submittedValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountStatusDataQualityIssue) MarshalJSON() ([]byte, error) {
	type noMethod AccountStatusDataQualityIssue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountStatusExampleItem: An example of an item that has poor data
// quality. An item value on the landing page differs from what is
// submitted, or conflicts with a policy.
type AccountStatusExampleItem struct {
	// ItemId: Unique item ID as specified in the uploaded product data.
	ItemId string `json:"itemId,omitempty"`

	// Link: Landing page of the item.
	Link string `json:"link,omitempty"`

	// SubmittedValue: The item value that was submitted.
	SubmittedValue string `json:"submittedValue,omitempty"`

	// Title: Title of the item.
	Title string `json:"title,omitempty"`

	// ValueOnLandingPage: The actual value on the landing page.
	ValueOnLandingPage string `json:"valueOnLandingPage,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ItemId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ItemId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountStatusExampleItem) MarshalJSON() ([]byte, error) {
	type noMethod AccountStatusExampleItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountTax: The tax settings of a merchant account.
type AccountTax struct {
	// AccountId: The ID of the account to which these account tax settings
	// belong.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountTax".
	Kind string `json:"kind,omitempty"`

	// Rules: Tax rules. Updating the tax rules will enable US taxes (not
	// reversible). Defining no rules is equivalent to not charging tax at
	// all.
	Rules []*AccountTaxTaxRule `json:"rules,omitempty"`

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

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountTax) MarshalJSON() ([]byte, error) {
	type noMethod AccountTax
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountTaxTaxRule: Tax calculation rule to apply in a state or
// province (USA only).
type AccountTaxTaxRule struct {
	// Country: Country code in which tax is applicable.
	Country string `json:"country,omitempty"`

	// LocationId: State (or province) is which the tax is applicable,
	// described by its location id (also called criteria id).
	LocationId uint64 `json:"locationId,omitempty,string"`

	// RatePercent: Explicit tax rate in percent, represented as a floating
	// point number without the percentage character. Must not be negative.
	RatePercent string `json:"ratePercent,omitempty"`

	// ShippingTaxed: If true, shipping charges are also taxed.
	ShippingTaxed bool `json:"shippingTaxed,omitempty"`

	// UseGlobalRate: Whether the tax rate is taken from a global tax table
	// or specified explicitly.
	UseGlobalRate bool `json:"useGlobalRate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountTaxTaxRule) MarshalJSON() ([]byte, error) {
	type noMethod AccountTaxTaxRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountUser struct {
	// Admin: Whether user is an admin.
	Admin *bool `json:"admin,omitempty"`

	// EmailAddress: User's email address.
	EmailAddress string `json:"emailAddress,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Admin") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Admin") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountUser) MarshalJSON() ([]byte, error) {
	type noMethod AccountUser
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountsAuthInfoResponse struct {
	// AccountIdentifiers: The account identifiers corresponding to the
	// authenticated user.
	// - For an individual account: only the merchant ID is defined
	// - For an aggregator: only the aggregator ID is defined
	// - For a subaccount of an MCA: both the merchant ID and the aggregator
	// ID are defined.
	AccountIdentifiers []*AccountIdentifier `json:"accountIdentifiers,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountsAuthInfoResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountIdentifiers")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountIdentifiers") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AccountsAuthInfoResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountsAuthInfoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountsClaimWebsiteResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountsClaimWebsiteResponse".
	Kind string `json:"kind,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountsClaimWebsiteResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountsClaimWebsiteResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountsCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*AccountsCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountsCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountsCustomBatchRequestEntry: A batch entry encoding a single
// non-batch accounts request.
type AccountsCustomBatchRequestEntry struct {
	// Account: The account to create or update. Only defined if the method
	// is insert or update.
	Account *Account `json:"account,omitempty"`

	// AccountId: The ID of the targeted account. Only defined if the method
	// is get, delete or claimwebsite.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// Force: Whether the account should be deleted if the account has
	// offers. Only applicable if the method is delete.
	Force bool `json:"force,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// Overwrite: Only applicable if the method is claimwebsite. Indicates
	// whether or not to take the claim from another account in case there
	// is a conflict.
	Overwrite bool `json:"overwrite,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Account") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Account") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountsCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountsCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*AccountsCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountsCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountsCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountsCustomBatchResponseEntry: A batch entry encoding a single
// non-batch accounts response.
type AccountsCustomBatchResponseEntry struct {
	// Account: The retrieved, created, or updated account. Not defined if
	// the method was delete or claimwebsite.
	Account *Account `json:"account,omitempty"`

	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountsCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Account") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Account") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountsCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountsListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountsListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// accounts.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*Account `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountstatusesCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*AccountstatusesCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountstatusesCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountstatusesCustomBatchRequestEntry: A batch entry encoding a
// single non-batch accountstatuses request.
type AccountstatusesCustomBatchRequestEntry struct {
	// AccountId: The ID of the (sub-)account whose status to get.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	// Method: The method (get).
	Method string `json:"method,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountstatusesCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountstatusesCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*AccountstatusesCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountstatusesCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountstatusesCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountstatusesCustomBatchResponseEntry: A batch entry encoding a
// single non-batch accountstatuses response.
type AccountstatusesCustomBatchResponseEntry struct {
	// AccountStatus: The requested account status. Defined if and only if
	// the request was successful.
	AccountStatus *AccountStatus `json:"accountStatus,omitempty"`

	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountStatus") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountstatusesCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccountstatusesListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountstatusesListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// account statuses.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*AccountStatus `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountstatusesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccounttaxCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*AccounttaxCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccounttaxCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccounttaxCustomBatchRequestEntry: A batch entry encoding a single
// non-batch accounttax request.
type AccounttaxCustomBatchRequestEntry struct {
	// AccountId: The ID of the account for which to get/update account tax
	// settings.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// AccountTax: The account tax settings to update. Only defined if the
	// method is update.
	AccountTax *AccountTax `json:"accountTax,omitempty"`

	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccounttaxCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccounttaxCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*AccounttaxCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accounttaxCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccounttaxCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccounttaxCustomBatchResponseEntry: A batch entry encoding a single
// non-batch accounttax response.
type AccounttaxCustomBatchResponseEntry struct {
	// AccountTax: The retrieved or updated account tax settings.
	AccountTax *AccountTax `json:"accountTax,omitempty"`

	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accounttaxCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountTax") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountTax") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccounttaxCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AccounttaxListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accounttaxListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// account tax settings.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*AccountTax `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccounttaxListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type CarrierRate struct {
	// CarrierName: Carrier service, such as "UPS" or "Fedex". The list of
	// supported carriers can be retrieved via the getSupportedCarriers
	// method. Required.
	CarrierName string `json:"carrierName,omitempty"`

	// CarrierService: Carrier service, such as "ground" or "2 days". The
	// list of supported services for a carrier can be retrieved via the
	// getSupportedCarriers method. Required.
	CarrierService string `json:"carrierService,omitempty"`

	// FlatAdjustment: Additive shipping rate modifier. Can be negative. For
	// example { "value": "1", "currency" : "USD" } adds $1 to the rate, {
	// "value": "-3", "currency" : "USD" } removes $3 from the rate.
	// Optional.
	FlatAdjustment *Price `json:"flatAdjustment,omitempty"`

	// Name: Name of the carrier rate. Must be unique per rate group.
	// Required.
	Name string `json:"name,omitempty"`

	// OriginPostalCode: Shipping origin for this carrier rate. Required.
	OriginPostalCode string `json:"originPostalCode,omitempty"`

	// PercentageAdjustment: Multiplicative shipping rate modifier as a
	// number in decimal notation. Can be negative. For example "5.4"
	// increases the rate by 5.4%, "-3" decreases the rate by 3%. Optional.
	PercentageAdjustment string `json:"percentageAdjustment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CarrierName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CarrierName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CarrierRate) MarshalJSON() ([]byte, error) {
	type noMethod CarrierRate
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type CarriersCarrier struct {
	// Country: The CLDR country code of the carrier (e.g., "US"). Always
	// present.
	Country string `json:"country,omitempty"`

	// Name: The name of the carrier (e.g., "UPS"). Always present.
	Name string `json:"name,omitempty"`

	// Services: A list of supported services (e.g., "ground") for that
	// carrier. Contains at least one service.
	Services []string `json:"services,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CarriersCarrier) MarshalJSON() ([]byte, error) {
	type noMethod CarriersCarrier
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Datafeed: Datafeed configuration data.
type Datafeed struct {
	// AttributeLanguage: The two-letter ISO 639-1 language in which the
	// attributes are defined in the data feed.
	AttributeLanguage string `json:"attributeLanguage,omitempty"`

	// ContentLanguage: The two-letter ISO 639-1 language of the items in
	// the feed. Must be a valid language for targetCountry.
	ContentLanguage string `json:"contentLanguage,omitempty"`

	// ContentType: The type of data feed. For product inventory feeds, only
	// feeds for local stores, not online stores, are supported.
	ContentType string `json:"contentType,omitempty"`

	// FetchSchedule: Fetch schedule for the feed file.
	FetchSchedule *DatafeedFetchSchedule `json:"fetchSchedule,omitempty"`

	// FileName: The filename of the feed. All feeds must have a unique file
	// name.
	FileName string `json:"fileName,omitempty"`

	// Format: Format of the feed file.
	Format *DatafeedFormat `json:"format,omitempty"`

	// Id: The ID of the data feed.
	Id int64 `json:"id,omitempty,string"`

	// IntendedDestinations: The list of intended destinations (corresponds
	// to checked check boxes in Merchant Center).
	IntendedDestinations []string `json:"intendedDestinations,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#datafeed".
	Kind string `json:"kind,omitempty"`

	// Name: A descriptive name of the data feed.
	Name string `json:"name,omitempty"`

	// TargetCountry: The country where the items in the feed will be
	// included in the search index, represented as a CLDR territory code.
	TargetCountry string `json:"targetCountry,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AttributeLanguage")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AttributeLanguage") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Datafeed) MarshalJSON() ([]byte, error) {
	type noMethod Datafeed
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedFetchSchedule: The required fields vary based on the
// frequency of fetching. For a monthly fetch schedule, day_of_month and
// hour are required. For a weekly fetch schedule, weekday and hour are
// required. For a daily fetch schedule, only hour is required.
type DatafeedFetchSchedule struct {
	// DayOfMonth: The day of the month the feed file should be fetched
	// (1-31).
	DayOfMonth int64 `json:"dayOfMonth,omitempty"`

	// FetchUrl: The URL where the feed file can be fetched. Google Merchant
	// Center will support automatic scheduled uploads using the HTTP,
	// HTTPS, FTP, or SFTP protocols, so the value will need to be a valid
	// link using one of those four protocols.
	FetchUrl string `json:"fetchUrl,omitempty"`

	// Hour: The hour of the day the feed file should be fetched (0-23).
	Hour int64 `json:"hour,omitempty"`

	// MinuteOfHour: The minute of the hour the feed file should be fetched
	// (0-59). Read-only.
	MinuteOfHour int64 `json:"minuteOfHour,omitempty"`

	// Password: An optional password for fetch_url.
	Password string `json:"password,omitempty"`

	// Paused: Whether the scheduled fetch is paused or not.
	Paused bool `json:"paused,omitempty"`

	// TimeZone: Time zone used for schedule. UTC by default. E.g.,
	// "America/Los_Angeles".
	TimeZone string `json:"timeZone,omitempty"`

	// Username: An optional user name for fetch_url.
	Username string `json:"username,omitempty"`

	// Weekday: The day of the week the feed file should be fetched.
	Weekday string `json:"weekday,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DayOfMonth") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DayOfMonth") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedFetchSchedule) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedFetchSchedule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatafeedFormat struct {
	// ColumnDelimiter: Delimiter for the separation of values in a
	// delimiter-separated values feed. If not specified, the delimiter will
	// be auto-detected. Ignored for non-DSV data feeds.
	ColumnDelimiter string `json:"columnDelimiter,omitempty"`

	// FileEncoding: Character encoding scheme of the data feed. If not
	// specified, the encoding will be auto-detected.
	FileEncoding string `json:"fileEncoding,omitempty"`

	// QuotingMode: Specifies how double quotes are interpreted. If not
	// specified, the mode will be auto-detected. Ignored for non-DSV data
	// feeds.
	QuotingMode string `json:"quotingMode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnDelimiter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnDelimiter") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedFormat) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedFormat
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedStatus: The status of a datafeed, i.e., the result of the
// last retrieval of the datafeed computed asynchronously when the feed
// processing is finished.
type DatafeedStatus struct {
	// DatafeedId: The ID of the feed for which the status is reported.
	DatafeedId uint64 `json:"datafeedId,omitempty,string"`

	// Errors: The list of errors occurring in the feed.
	Errors []*DatafeedStatusError `json:"errors,omitempty"`

	// ItemsTotal: The number of items in the feed that were processed.
	ItemsTotal uint64 `json:"itemsTotal,omitempty,string"`

	// ItemsValid: The number of items in the feed that were valid.
	ItemsValid uint64 `json:"itemsValid,omitempty,string"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#datafeedStatus".
	Kind string `json:"kind,omitempty"`

	// LastUploadDate: The last date at which the feed was uploaded.
	LastUploadDate string `json:"lastUploadDate,omitempty"`

	// ProcessingStatus: The processing status of the feed.
	ProcessingStatus string `json:"processingStatus,omitempty"`

	// Warnings: The list of errors occurring in the feed.
	Warnings []*DatafeedStatusError `json:"warnings,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DatafeedId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatafeedId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedStatus) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedStatusError: An error occurring in the feed, like "invalid
// price".
type DatafeedStatusError struct {
	// Code: The code of the error, e.g., "validation/invalid_value".
	Code string `json:"code,omitempty"`

	// Count: The number of occurrences of the error in the feed.
	Count uint64 `json:"count,omitempty,string"`

	// Examples: A list of example occurrences of the error, grouped by
	// product.
	Examples []*DatafeedStatusExample `json:"examples,omitempty"`

	// Message: The error message, e.g., "Invalid price".
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedStatusError) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedStatusError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedStatusExample: An example occurrence for a particular error.
type DatafeedStatusExample struct {
	// ItemId: The ID of the example item.
	ItemId string `json:"itemId,omitempty"`

	// LineNumber: Line number in the data feed where the example is found.
	LineNumber uint64 `json:"lineNumber,omitempty,string"`

	// Value: The problematic value.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ItemId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ItemId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedStatusExample) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedStatusExample
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatafeedsCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*DatafeedsCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedsCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedsCustomBatchRequestEntry: A batch entry encoding a single
// non-batch datafeeds request.
type DatafeedsCustomBatchRequestEntry struct {
	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// Datafeed: The data feed to insert.
	Datafeed *Datafeed `json:"datafeed,omitempty"`

	// DatafeedId: The ID of the data feed to get or delete.
	DatafeedId uint64 `json:"datafeedId,omitempty,string"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedsCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatafeedsCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*DatafeedsCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#datafeedsCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedsCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedsCustomBatchResponseEntry: A batch entry encoding a single
// non-batch datafeeds response.
type DatafeedsCustomBatchResponseEntry struct {
	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Datafeed: The requested data feed. Defined if and only if the request
	// was successful.
	Datafeed *Datafeed `json:"datafeed,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedsCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatafeedsListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#datafeedsListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// datafeeds.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*Datafeed `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatafeedstatusesCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*DatafeedstatusesCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedstatusesCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedstatusesCustomBatchRequestEntry: A batch entry encoding a
// single non-batch datafeedstatuses request.
type DatafeedstatusesCustomBatchRequestEntry struct {
	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// DatafeedId: The ID of the data feed to get or delete.
	DatafeedId uint64 `json:"datafeedId,omitempty,string"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedstatusesCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatafeedstatusesCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*DatafeedstatusesCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#datafeedstatusesCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedstatusesCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatafeedstatusesCustomBatchResponseEntry: A batch entry encoding a
// single non-batch datafeedstatuses response.
type DatafeedstatusesCustomBatchResponseEntry struct {
	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// DatafeedStatus: The requested data feed status. Defined if and only
	// if the request was successful.
	DatafeedStatus *DatafeedStatus `json:"datafeedStatus,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedstatusesCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatafeedstatusesListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#datafeedstatusesListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// datafeed statuses.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*DatafeedStatus `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatafeedstatusesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DeliveryTime struct {
	// MaxTransitTimeInDays: Maximum number of business days that is spent
	// in transit. 0 means same day delivery, 1 means next day delivery.
	// Must be greater than or equal to minTransitTimeInDays. Required.
	MaxTransitTimeInDays int64 `json:"maxTransitTimeInDays,omitempty"`

	// MinTransitTimeInDays: Minimum number of business days that is spent
	// in transit. 0 means same day delivery, 1 means next day delivery.
	// Required.
	MinTransitTimeInDays int64 `json:"minTransitTimeInDays,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "MaxTransitTimeInDays") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxTransitTimeInDays") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DeliveryTime) MarshalJSON() ([]byte, error) {
	type noMethod DeliveryTime
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Error: An error returned by the API.
type Error struct {
	// Domain: The domain of the error.
	Domain string `json:"domain,omitempty"`

	// Message: A description of the error.
	Message string `json:"message,omitempty"`

	// Reason: The error code.
	Reason string `json:"reason,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Domain") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Domain") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Error) MarshalJSON() ([]byte, error) {
	type noMethod Error
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Errors: A list of errors returned by a failed batch entry.
type Errors struct {
	// Code: The HTTP status of the first error in errors.
	Code int64 `json:"code,omitempty"`

	// Errors: A list of errors.
	Errors []*Error `json:"errors,omitempty"`

	// Message: The message of the first error in errors.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Errors) MarshalJSON() ([]byte, error) {
	type noMethod Errors
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Headers: A non-empty list of row or column headers for a table.
// Exactly one of prices, weights, numItems, postalCodeGroupNames, or
// locations must be set.
type Headers struct {
	// Locations: A list of location ID sets. Must be non-empty. Can only be
	// set if all other fields are not set.
	Locations []*LocationIdSet `json:"locations,omitempty"`

	// NumberOfItems: A list of inclusive number of items upper bounds. The
	// last value can be "infinity". For example ["10", "50", "infinity"]
	// represents the headers "<= 10 items", " 50 items". Must be non-empty.
	// Can only be set if all other fields are not set.
	NumberOfItems []string `json:"numberOfItems,omitempty"`

	// PostalCodeGroupNames: A list of postal group names. The last value
	// can be "all other locations". Example: ["zone 1", "zone 2", "all
	// other locations"]. The referred postal code groups must match the
	// delivery country of the service. Must be non-empty. Can only be set
	// if all other fields are not set.
	PostalCodeGroupNames []string `json:"postalCodeGroupNames,omitempty"`

	// Prices: be "infinity". For example [{"value": "10", "currency":
	// "USD"}, {"value": "500", "currency": "USD"}, {"value": "infinity",
	// "currency": "USD"}] represents the headers "<= $10", " $500". All
	// prices within a service must have the same currency. Must be
	// non-empty. Can only be set if all other fields are not set.
	Prices []*Price `json:"prices,omitempty"`

	// Weights: be "infinity". For example [{"value": "10", "unit": "kg"},
	// {"value": "50", "unit": "kg"}, {"value": "infinity", "unit": "kg"}]
	// represents the headers "<= 10kg", " 50kg". All weights within a
	// service must have the same unit. Must be non-empty. Can only be set
	// if all other fields are not set.
	Weights []*Weight `json:"weights,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Locations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Locations") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Headers) MarshalJSON() ([]byte, error) {
	type noMethod Headers
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Installment struct {
	// Amount: The amount the buyer has to pay per month.
	Amount *Price `json:"amount,omitempty"`

	// Months: The number of installments the buyer has to pay.
	Months int64 `json:"months,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Amount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Amount") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Installment) MarshalJSON() ([]byte, error) {
	type noMethod Installment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Inventory struct {
	// Availability: The availability of the product.
	Availability string `json:"availability,omitempty"`

	// Installment: Number and amount of installments to pay for an item.
	// Brazil only.
	Installment *Installment `json:"installment,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#inventory".
	Kind string `json:"kind,omitempty"`

	// LoyaltyPoints: Loyalty points that users receive after purchasing the
	// item. Japan only.
	LoyaltyPoints *LoyaltyPoints `json:"loyaltyPoints,omitempty"`

	// Pickup: Store pickup information. Only supported for local inventory.
	// Not setting pickup means "don't update" while setting it to the empty
	// value ({} in JSON) means "delete". Otherwise, pickupMethod and
	// pickupSla must be set together, unless pickupMethod is "not
	// supported".
	Pickup *InventoryPickup `json:"pickup,omitempty"`

	// Price: The price of the product.
	Price *Price `json:"price,omitempty"`

	// Quantity: The quantity of the product. Must be equal to or greater
	// than zero. Supported only for local products.
	Quantity int64 `json:"quantity,omitempty"`

	// SalePrice: The sale price of the product. Mandatory if
	// sale_price_effective_date is defined.
	SalePrice *Price `json:"salePrice,omitempty"`

	// SalePriceEffectiveDate: A date range represented by a pair of ISO
	// 8601 dates separated by a space, comma, or slash. Both dates might be
	// specified as 'null' if undecided.
	SalePriceEffectiveDate string `json:"salePriceEffectiveDate,omitempty"`

	// SellOnGoogleQuantity: The quantity of the product that is reserved
	// for sell-on-google ads. Supported only for online products.
	SellOnGoogleQuantity int64 `json:"sellOnGoogleQuantity,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Availability") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Availability") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Inventory) MarshalJSON() ([]byte, error) {
	type noMethod Inventory
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InventoryCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*InventoryCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InventoryCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InventoryCustomBatchRequestEntry: A batch entry encoding a single
// non-batch inventory request.
type InventoryCustomBatchRequestEntry struct {
	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// Inventory: Price and availability of the product.
	Inventory *Inventory `json:"inventory,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	// ProductId: The ID of the product for which to update price and
	// availability.
	ProductId string `json:"productId,omitempty"`

	// StoreCode: The code of the store for which to update price and
	// availability. Use online to update price and availability of an
	// online product.
	StoreCode string `json:"storeCode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InventoryCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InventoryCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*InventoryCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#inventoryCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InventoryCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InventoryCustomBatchResponseEntry: A batch entry encoding a single
// non-batch inventory response.
type InventoryCustomBatchResponseEntry struct {
	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#inventoryCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InventoryCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InventoryPickup struct {
	// PickupMethod: Whether store pickup is available for this offer and
	// whether the pickup option should be shown as buy, reserve, or not
	// supported. Only supported for local inventory. Unless the value is
	// "not supported", must be submitted together with pickupSla.
	PickupMethod string `json:"pickupMethod,omitempty"`

	// PickupSla: The expected date that an order will be ready for pickup,
	// relative to when the order is placed. Only supported for local
	// inventory. Must be submitted together with pickupMethod.
	PickupSla string `json:"pickupSla,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PickupMethod") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PickupMethod") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InventoryPickup) MarshalJSON() ([]byte, error) {
	type noMethod InventoryPickup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InventorySetRequest struct {
	// Availability: The availability of the product.
	Availability string `json:"availability,omitempty"`

	// Installment: Number and amount of installments to pay for an item.
	// Brazil only.
	Installment *Installment `json:"installment,omitempty"`

	// LoyaltyPoints: Loyalty points that users receive after purchasing the
	// item. Japan only.
	LoyaltyPoints *LoyaltyPoints `json:"loyaltyPoints,omitempty"`

	// Pickup: Store pickup information. Only supported for local inventory.
	// Not setting pickup means "don't update" while setting it to the empty
	// value ({} in JSON) means "delete". Otherwise, pickupMethod and
	// pickupSla must be set together, unless pickupMethod is "not
	// supported".
	Pickup *InventoryPickup `json:"pickup,omitempty"`

	// Price: The price of the product.
	Price *Price `json:"price,omitempty"`

	// Quantity: The quantity of the product. Must be equal to or greater
	// than zero. Supported only for local products.
	Quantity int64 `json:"quantity,omitempty"`

	// SalePrice: The sale price of the product. Mandatory if
	// sale_price_effective_date is defined.
	SalePrice *Price `json:"salePrice,omitempty"`

	// SalePriceEffectiveDate: A date range represented by a pair of ISO
	// 8601 dates separated by a space, comma, or slash. Both dates might be
	// specified as 'null' if undecided.
	SalePriceEffectiveDate string `json:"salePriceEffectiveDate,omitempty"`

	// SellOnGoogleQuantity: The quantity of the product that is reserved
	// for sell-on-google ads. Supported only for online products.
	SellOnGoogleQuantity int64 `json:"sellOnGoogleQuantity,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Availability") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Availability") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InventorySetRequest) MarshalJSON() ([]byte, error) {
	type noMethod InventorySetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InventorySetResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#inventorySetResponse".
	Kind string `json:"kind,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InventorySetResponse) MarshalJSON() ([]byte, error) {
	type noMethod InventorySetResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type LocationIdSet struct {
	// LocationIds: A non-empty list of location IDs. They must all be of
	// the same location type (e.g., state).
	LocationIds []string `json:"locationIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LocationIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LocationIds") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LocationIdSet) MarshalJSON() ([]byte, error) {
	type noMethod LocationIdSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type LoyaltyPoints struct {
	// Name: Name of loyalty points program. It is recommended to limit the
	// name to 12 full-width characters or 24 Roman characters.
	Name string `json:"name,omitempty"`

	// PointsValue: The retailer's loyalty points in absolute value.
	PointsValue int64 `json:"pointsValue,omitempty,string"`

	// Ratio: The ratio of a point when converted to currency. Google
	// assumes currency based on Merchant Center settings. If ratio is left
	// out, it defaults to 1.0.
	Ratio float64 `json:"ratio,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Name") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LoyaltyPoints) MarshalJSON() ([]byte, error) {
	type noMethod LoyaltyPoints
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *LoyaltyPoints) UnmarshalJSON(data []byte) error {
	type noMethod LoyaltyPoints
	var s1 struct {
		Ratio gensupport.JSONFloat64 `json:"ratio"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Ratio = float64(s1.Ratio)
	return nil
}

type Order struct {
	// Acknowledged: Whether the order was acknowledged.
	Acknowledged bool `json:"acknowledged,omitempty"`

	// ChannelType: The channel type of the order: "purchaseOnGoogle" or
	// "googleExpress".
	ChannelType string `json:"channelType,omitempty"`

	// Customer: The details of the customer who placed the order.
	Customer *OrderCustomer `json:"customer,omitempty"`

	// DeliveryDetails: The details for the delivery.
	DeliveryDetails *OrderDeliveryDetails `json:"deliveryDetails,omitempty"`

	// Id: The REST id of the order. Globally unique.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#order".
	Kind string `json:"kind,omitempty"`

	// LineItems: Line items that are ordered.
	LineItems []*OrderLineItem `json:"lineItems,omitempty"`

	MerchantId uint64 `json:"merchantId,omitempty,string"`

	// MerchantOrderId: Merchant-provided id of the order.
	MerchantOrderId string `json:"merchantOrderId,omitempty"`

	// NetAmount: The net amount for the order. For example, if an order was
	// originally for a grand total of $100 and a refund was issued for $20,
	// the net amount will be $80.
	NetAmount *Price `json:"netAmount,omitempty"`

	// PaymentMethod: The details of the payment method.
	PaymentMethod *OrderPaymentMethod `json:"paymentMethod,omitempty"`

	// PaymentStatus: The status of the payment.
	PaymentStatus string `json:"paymentStatus,omitempty"`

	// PlacedDate: The date when the order was placed, in ISO 8601 format.
	PlacedDate string `json:"placedDate,omitempty"`

	// Promotions: The details of the merchant provided promotions applied
	// to the order. More details about the program are here.
	Promotions []*OrderPromotion `json:"promotions,omitempty"`

	// Refunds: Refunds for the order.
	Refunds []*OrderRefund `json:"refunds,omitempty"`

	// Shipments: Shipments of the order.
	Shipments []*OrderShipment `json:"shipments,omitempty"`

	// ShippingCost: The total cost of shipping for all items.
	ShippingCost *Price `json:"shippingCost,omitempty"`

	// ShippingCostTax: The tax for the total shipping cost.
	ShippingCostTax *Price `json:"shippingCostTax,omitempty"`

	// ShippingOption: The requested shipping option.
	ShippingOption string `json:"shippingOption,omitempty"`

	// Status: The status of the order.
	Status string `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Acknowledged") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Acknowledged") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Order) MarshalJSON() ([]byte, error) {
	type noMethod Order
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderAddress struct {
	// Country: CLDR country code (e.g. "US").
	Country string `json:"country,omitempty"`

	// FullAddress: Strings representing the lines of the printed label for
	// mailing the order, for example:
	// John Smith
	// 1600 Amphitheatre Parkway
	// Mountain View, CA, 94043
	// United States
	FullAddress []string `json:"fullAddress,omitempty"`

	// IsPostOfficeBox: Whether the address is a post office box.
	IsPostOfficeBox bool `json:"isPostOfficeBox,omitempty"`

	// Locality: City, town or commune. May also include dependent
	// localities or sublocalities (e.g. neighborhoods or suburbs).
	Locality string `json:"locality,omitempty"`

	// PostalCode: Postal Code or ZIP (e.g. "94043").
	PostalCode string `json:"postalCode,omitempty"`

	// RecipientName: Name of the recipient.
	RecipientName string `json:"recipientName,omitempty"`

	// Region: Top-level administrative subdivision of the country (e.g.
	// "CA").
	Region string `json:"region,omitempty"`

	// StreetAddress: Street-level part of the address.
	StreetAddress []string `json:"streetAddress,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderAddress) MarshalJSON() ([]byte, error) {
	type noMethod OrderAddress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderCancellation struct {
	// Actor: The actor that created the cancellation.
	Actor string `json:"actor,omitempty"`

	// CreationDate: Date on which the cancellation has been created, in ISO
	// 8601 format.
	CreationDate string `json:"creationDate,omitempty"`

	// Quantity: The quantity that was canceled.
	Quantity int64 `json:"quantity,omitempty"`

	// Reason: The reason for the cancellation. Orders that are cancelled
	// with a noInventory reason will lead to the removal of the product
	// from POG until you make an update to that product. This will not
	// affect your Shopping ads.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Actor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Actor") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderCancellation) MarshalJSON() ([]byte, error) {
	type noMethod OrderCancellation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderCustomer struct {
	// Email: Email address of the customer.
	Email string `json:"email,omitempty"`

	// ExplicitMarketingPreference: If set, this indicates the user
	// explicitly chose to opt in or out of providing marketing rights to
	// the merchant. If unset, this indicates the user has already made this
	// choice in a previous purchase, and was thus not shown the marketing
	// right opt in/out checkbox during the checkout flow.
	ExplicitMarketingPreference bool `json:"explicitMarketingPreference,omitempty"`

	// FullName: Full name of the customer.
	FullName string `json:"fullName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Email") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Email") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderCustomer) MarshalJSON() ([]byte, error) {
	type noMethod OrderCustomer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderDeliveryDetails struct {
	// Address: The delivery address
	Address *OrderAddress `json:"address,omitempty"`

	// PhoneNumber: The phone number of the person receiving the delivery.
	PhoneNumber string `json:"phoneNumber,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Address") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Address") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderDeliveryDetails) MarshalJSON() ([]byte, error) {
	type noMethod OrderDeliveryDetails
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderLineItem struct {
	// Cancellations: Cancellations of the line item.
	Cancellations []*OrderCancellation `json:"cancellations,omitempty"`

	// Id: The id of the line item.
	Id string `json:"id,omitempty"`

	// Price: Total price for the line item. For example, if two items for
	// $10 are purchased, the total price will be $20.
	Price *Price `json:"price,omitempty"`

	// Product: Product data from the time of the order placement.
	Product *OrderLineItemProduct `json:"product,omitempty"`

	// QuantityCanceled: Number of items canceled.
	QuantityCanceled int64 `json:"quantityCanceled,omitempty"`

	// QuantityDelivered: Number of items delivered.
	QuantityDelivered int64 `json:"quantityDelivered,omitempty"`

	// QuantityOrdered: Number of items ordered.
	QuantityOrdered int64 `json:"quantityOrdered,omitempty"`

	// QuantityPending: Number of items pending.
	QuantityPending int64 `json:"quantityPending,omitempty"`

	// QuantityReturned: Number of items returned.
	QuantityReturned int64 `json:"quantityReturned,omitempty"`

	// QuantityShipped: Number of items shipped.
	QuantityShipped int64 `json:"quantityShipped,omitempty"`

	// ReturnInfo: Details of the return policy for the line item.
	ReturnInfo *OrderLineItemReturnInfo `json:"returnInfo,omitempty"`

	// Returns: Returns of the line item.
	Returns []*OrderReturn `json:"returns,omitempty"`

	// ShippingDetails: Details of the requested shipping for the line item.
	ShippingDetails *OrderLineItemShippingDetails `json:"shippingDetails,omitempty"`

	// Tax: Total tax amount for the line item. For example, if two items
	// are purchased, and each have a cost tax of $2, the total tax amount
	// will be $4.
	Tax *Price `json:"tax,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cancellations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cancellations") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderLineItem) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderLineItemProduct struct {
	// Brand: Brand of the item.
	Brand string `json:"brand,omitempty"`

	// Channel: The item's channel (online or local).
	Channel string `json:"channel,omitempty"`

	// Condition: Condition or state of the item.
	Condition string `json:"condition,omitempty"`

	// ContentLanguage: The two-letter ISO 639-1 language code for the item.
	ContentLanguage string `json:"contentLanguage,omitempty"`

	// Gtin: Global Trade Item Number (GTIN) of the item.
	Gtin string `json:"gtin,omitempty"`

	// Id: The REST id of the product.
	Id string `json:"id,omitempty"`

	// ImageLink: URL of an image of the item.
	ImageLink string `json:"imageLink,omitempty"`

	// ItemGroupId: Shared identifier for all variants of the same product.
	ItemGroupId string `json:"itemGroupId,omitempty"`

	// Mpn: Manufacturer Part Number (MPN) of the item.
	Mpn string `json:"mpn,omitempty"`

	// OfferId: An identifier of the item.
	OfferId string `json:"offerId,omitempty"`

	// Price: Price of the item.
	Price *Price `json:"price,omitempty"`

	// ShownImage: URL to the cached image shown to the user when order was
	// placed.
	ShownImage string `json:"shownImage,omitempty"`

	// TargetCountry: The CLDR territory code of the target country of the
	// product.
	TargetCountry string `json:"targetCountry,omitempty"`

	// Title: The title of the product.
	Title string `json:"title,omitempty"`

	// VariantAttributes: Variant attributes for the item. These are
	// dimensions of the product, such as color, gender, material, pattern,
	// and size. You can find a comprehensive list of variant attributes
	// here.
	VariantAttributes []*OrderLineItemProductVariantAttribute `json:"variantAttributes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Brand") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Brand") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderLineItemProduct) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemProduct
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderLineItemProductVariantAttribute struct {
	// Dimension: The dimension of the variant.
	Dimension string `json:"dimension,omitempty"`

	// Value: The value for the dimension.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Dimension") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Dimension") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderLineItemProductVariantAttribute) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemProductVariantAttribute
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderLineItemReturnInfo struct {
	// DaysToReturn: How many days later the item can be returned.
	DaysToReturn int64 `json:"daysToReturn,omitempty"`

	// IsReturnable: Whether the item is returnable.
	IsReturnable bool `json:"isReturnable,omitempty"`

	// PolicyUrl: URL of the item return policy.
	PolicyUrl string `json:"policyUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DaysToReturn") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DaysToReturn") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderLineItemReturnInfo) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemReturnInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderLineItemShippingDetails struct {
	// DeliverByDate: The delivery by date, in ISO 8601 format.
	DeliverByDate string `json:"deliverByDate,omitempty"`

	// Method: Details of the shipping method.
	Method *OrderLineItemShippingDetailsMethod `json:"method,omitempty"`

	// ShipByDate: The ship by date, in ISO 8601 format.
	ShipByDate string `json:"shipByDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeliverByDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeliverByDate") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderLineItemShippingDetails) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemShippingDetails
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderLineItemShippingDetailsMethod struct {
	// Carrier: The carrier for the shipping. Optional.
	Carrier string `json:"carrier,omitempty"`

	// MaxDaysInTransit: Maximum transit time.
	MaxDaysInTransit int64 `json:"maxDaysInTransit,omitempty"`

	// MethodName: The name of the shipping method.
	MethodName string `json:"methodName,omitempty"`

	// MinDaysInTransit: Minimum transit time.
	MinDaysInTransit int64 `json:"minDaysInTransit,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carrier") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderLineItemShippingDetailsMethod) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemShippingDetailsMethod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderPaymentMethod struct {
	// BillingAddress: The billing address.
	BillingAddress *OrderAddress `json:"billingAddress,omitempty"`

	// ExpirationMonth: The card expiration month (January = 1, February = 2
	// etc.).
	ExpirationMonth int64 `json:"expirationMonth,omitempty"`

	// ExpirationYear: The card expiration year (4-digit, e.g. 2015).
	ExpirationYear int64 `json:"expirationYear,omitempty"`

	// LastFourDigits: The last four digits of the card number.
	LastFourDigits string `json:"lastFourDigits,omitempty"`

	// PhoneNumber: The billing phone number.
	PhoneNumber string `json:"phoneNumber,omitempty"`

	// Type: The type of instrument (VISA, Mastercard, etc).
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BillingAddress") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BillingAddress") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrderPaymentMethod) MarshalJSON() ([]byte, error) {
	type noMethod OrderPaymentMethod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderPromotion struct {
	Benefits []*OrderPromotionBenefit `json:"benefits,omitempty"`

	// EffectiveDates: The date and time frame when the promotion is active
	// and ready for validation review. Note that the promotion live time
	// may be delayed for a few hours due to the validation review.
	// Start date and end date are separated by a forward slash (/). The
	// start date is specified by the format (YYYY-MM-DD), followed by the
	// letter ?T?, the time of the day when the sale starts (in Greenwich
	// Mean Time, GMT), followed by an expression of the time zone for the
	// sale. The end date is in the same format.
	EffectiveDates string `json:"effectiveDates,omitempty"`

	// GenericRedemptionCode: Optional. The text code that corresponds to
	// the promotion when applied on the retailer?s website.
	GenericRedemptionCode string `json:"genericRedemptionCode,omitempty"`

	// Id: The unique ID of the promotion.
	Id string `json:"id,omitempty"`

	// LongTitle: The full title of the promotion.
	LongTitle string `json:"longTitle,omitempty"`

	// ProductApplicability: Whether the promotion is applicable to all
	// products or only specific products.
	ProductApplicability string `json:"productApplicability,omitempty"`

	// RedemptionChannel: Indicates that the promotion is valid online.
	RedemptionChannel string `json:"redemptionChannel,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Benefits") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Benefits") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderPromotion) MarshalJSON() ([]byte, error) {
	type noMethod OrderPromotion
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderPromotionBenefit struct {
	// Discount: The discount in the order price when the promotion is
	// applied.
	Discount *Price `json:"discount,omitempty"`

	// OfferIds: The OfferId(s) that were purchased in this order and map to
	// this specific benefit of the promotion.
	OfferIds []string `json:"offerIds,omitempty"`

	// SubType: Further describes the benefit of the promotion. Note that we
	// will expand on this enumeration as we support new promotion
	// sub-types.
	SubType string `json:"subType,omitempty"`

	// TaxImpact: The impact on tax when the promotion is applied.
	TaxImpact *Price `json:"taxImpact,omitempty"`

	// Type: Describes whether the promotion applies to products (e.g. 20%
	// off) or to shipping (e.g. Free Shipping).
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Discount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Discount") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderPromotionBenefit) MarshalJSON() ([]byte, error) {
	type noMethod OrderPromotionBenefit
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderRefund struct {
	// Actor: The actor that created the refund.
	Actor string `json:"actor,omitempty"`

	// Amount: The amount that is refunded.
	Amount *Price `json:"amount,omitempty"`

	// CreationDate: Date on which the item has been created, in ISO 8601
	// format.
	CreationDate string `json:"creationDate,omitempty"`

	// Reason: The reason for the refund.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Actor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Actor") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderRefund) MarshalJSON() ([]byte, error) {
	type noMethod OrderRefund
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderReturn struct {
	// Actor: The actor that created the refund.
	Actor string `json:"actor,omitempty"`

	// CreationDate: Date on which the item has been created, in ISO 8601
	// format.
	CreationDate string `json:"creationDate,omitempty"`

	// Quantity: Quantity that is returned.
	Quantity int64 `json:"quantity,omitempty"`

	// Reason: The reason for the return.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Actor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Actor") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderReturn) MarshalJSON() ([]byte, error) {
	type noMethod OrderReturn
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderShipment struct {
	// Carrier: The carrier handling the shipment.
	Carrier string `json:"carrier,omitempty"`

	// CreationDate: Date on which the shipment has been created, in ISO
	// 8601 format.
	CreationDate string `json:"creationDate,omitempty"`

	// DeliveryDate: Date on which the shipment has been delivered, in ISO
	// 8601 format. Present only if status is delievered
	DeliveryDate string `json:"deliveryDate,omitempty"`

	// Id: The id of the shipment.
	Id string `json:"id,omitempty"`

	// LineItems: The line items that are shipped.
	LineItems []*OrderShipmentLineItemShipment `json:"lineItems,omitempty"`

	// Status: The status of the shipment.
	Status string `json:"status,omitempty"`

	// TrackingId: The tracking id for the shipment.
	TrackingId string `json:"trackingId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carrier") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderShipment) MarshalJSON() ([]byte, error) {
	type noMethod OrderShipment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrderShipmentLineItemShipment struct {
	// LineItemId: The id of the line item that is shipped.
	LineItemId string `json:"lineItemId,omitempty"`

	// Quantity: The quantity that is shipped.
	Quantity int64 `json:"quantity,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LineItemId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LineItemId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderShipmentLineItemShipment) MarshalJSON() ([]byte, error) {
	type noMethod OrderShipmentLineItemShipment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersAcknowledgeRequest struct {
	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersAcknowledgeRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersAcknowledgeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersAcknowledgeResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersAcknowledgeResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersAcknowledgeResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersAcknowledgeResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersAdvanceTestOrderResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersAdvanceTestOrderResponse".
	Kind string `json:"kind,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersAdvanceTestOrderResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersAdvanceTestOrderResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCancelLineItemRequest struct {
	// Amount: Amount to refund for the cancelation. Optional. If not set,
	// Google will calculate the default based on the price and tax of the
	// items involved. The amount must not be larger than the net amount
	// left on the order.
	Amount *Price `json:"amount,omitempty"`

	// LineItemId: The ID of the line item to cancel.
	LineItemId string `json:"lineItemId,omitempty"`

	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// Quantity: The quantity to cancel.
	Quantity int64 `json:"quantity,omitempty"`

	// Reason: The reason for the cancellation.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Amount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Amount") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCancelLineItemRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelLineItemRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCancelLineItemResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersCancelLineItemResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCancelLineItemResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelLineItemResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCancelRequest struct {
	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// Reason: The reason for the cancellation.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCancelRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCancelResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersCancelResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCancelResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCreateTestOrderRequest struct {
	// TemplateName: The test order template to use. Specify as an
	// alternative to testOrder as a shortcut for retrieving a template and
	// then creating an order using that template.
	TemplateName string `json:"templateName,omitempty"`

	// TestOrder: The test order to create.
	TestOrder *TestOrder `json:"testOrder,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TemplateName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TemplateName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCreateTestOrderRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCreateTestOrderRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCreateTestOrderResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersCreateTestOrderResponse".
	Kind string `json:"kind,omitempty"`

	// OrderId: The ID of the newly created test order.
	OrderId string `json:"orderId,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCreateTestOrderResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCreateTestOrderResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*OrdersCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequestEntry struct {
	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// Cancel: Required for cancel method.
	Cancel *OrdersCustomBatchRequestEntryCancel `json:"cancel,omitempty"`

	// CancelLineItem: Required for cancelLineItem method.
	CancelLineItem *OrdersCustomBatchRequestEntryCancelLineItem `json:"cancelLineItem,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	// MerchantOrderId: The merchant order id. Required for
	// updateMerchantOrderId and getByMerchantOrderId methods.
	MerchantOrderId string `json:"merchantOrderId,omitempty"`

	// Method: The method to apply.
	Method string `json:"method,omitempty"`

	// OperationId: The ID of the operation. Unique across all operations
	// for a given order. Required for all methods beside get and
	// getByMerchantOrderId.
	OperationId string `json:"operationId,omitempty"`

	// OrderId: The ID of the order. Required for all methods beside
	// getByMerchantOrderId.
	OrderId string `json:"orderId,omitempty"`

	// Refund: Required for refund method.
	Refund *OrdersCustomBatchRequestEntryRefund `json:"refund,omitempty"`

	// ReturnLineItem: Required for returnLineItem method.
	ReturnLineItem *OrdersCustomBatchRequestEntryReturnLineItem `json:"returnLineItem,omitempty"`

	// ShipLineItems: Required for shipLineItems method.
	ShipLineItems *OrdersCustomBatchRequestEntryShipLineItems `json:"shipLineItems,omitempty"`

	// UpdateShipment: Required for updateShipment method.
	UpdateShipment *OrdersCustomBatchRequestEntryUpdateShipment `json:"updateShipment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequestEntryCancel struct {
	// Reason: The reason for the cancellation.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Reason") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Reason") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequestEntryCancel) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryCancel
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequestEntryCancelLineItem struct {
	// Amount: Amount to refund for the cancelation. Optional. If not set,
	// Google will calculate the default based on the price and tax of the
	// items involved. The amount must not be larger than the net amount
	// left on the order.
	Amount *Price `json:"amount,omitempty"`

	// LineItemId: The ID of the line item to cancel.
	LineItemId string `json:"lineItemId,omitempty"`

	// Quantity: The quantity to cancel.
	Quantity int64 `json:"quantity,omitempty"`

	// Reason: The reason for the cancellation.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Amount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Amount") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequestEntryCancelLineItem) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryCancelLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequestEntryRefund struct {
	// Amount: The amount that is refunded.
	Amount *Price `json:"amount,omitempty"`

	// Reason: The reason for the refund.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Amount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Amount") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequestEntryRefund) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryRefund
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequestEntryReturnLineItem struct {
	// LineItemId: The ID of the line item to return.
	LineItemId string `json:"lineItemId,omitempty"`

	// Quantity: The quantity to return.
	Quantity int64 `json:"quantity,omitempty"`

	// Reason: The reason for the return.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LineItemId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LineItemId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequestEntryReturnLineItem) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryReturnLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequestEntryShipLineItems struct {
	// Carrier: The carrier handling the shipment.
	Carrier string `json:"carrier,omitempty"`

	// LineItems: Line items to ship.
	LineItems []*OrderShipmentLineItemShipment `json:"lineItems,omitempty"`

	// ShipmentId: The ID of the shipment.
	ShipmentId string `json:"shipmentId,omitempty"`

	// TrackingId: The tracking id for the shipment.
	TrackingId string `json:"trackingId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carrier") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequestEntryShipLineItems) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryShipLineItems
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchRequestEntryUpdateShipment struct {
	// Carrier: The carrier handling the shipment. Not updated if missing.
	Carrier string `json:"carrier,omitempty"`

	// ShipmentId: The ID of the shipment.
	ShipmentId string `json:"shipmentId,omitempty"`

	// Status: New status for the shipment. Not updated if missing.
	Status string `json:"status,omitempty"`

	// TrackingId: The tracking id for the shipment. Not updated if missing.
	TrackingId string `json:"trackingId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carrier") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchRequestEntryUpdateShipment) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryUpdateShipment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*OrdersCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersCustomBatchResponseEntry struct {
	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// ExecutionStatus: The status of the execution. Only defined if the
	// method is not get or getByMerchantOrderId and if the request was
	// successful.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// Order: The retrieved order. Only defined if the method is get and if
	// the request was successful.
	Order *Order `json:"order,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersGetByMerchantOrderIdResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersGetByMerchantOrderIdResponse".
	Kind string `json:"kind,omitempty"`

	// Order: The requested order.
	Order *Order `json:"order,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersGetByMerchantOrderIdResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersGetByMerchantOrderIdResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersGetTestOrderTemplateResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersGetTestOrderTemplateResponse".
	Kind string `json:"kind,omitempty"`

	// Template: The requested test order template.
	Template *TestOrder `json:"template,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersGetTestOrderTemplateResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersGetTestOrderTemplateResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// orders.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*Order `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersRefundRequest struct {
	// Amount: The amount that is refunded.
	Amount *Price `json:"amount,omitempty"`

	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// Reason: The reason for the refund.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Amount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Amount") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersRefundRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersRefundRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersRefundResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersRefundResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersRefundResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersRefundResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersReturnLineItemRequest struct {
	// LineItemId: The ID of the line item to return.
	LineItemId string `json:"lineItemId,omitempty"`

	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// Quantity: The quantity to return.
	Quantity int64 `json:"quantity,omitempty"`

	// Reason: The reason for the return.
	Reason string `json:"reason,omitempty"`

	// ReasonText: The explanation of the reason.
	ReasonText string `json:"reasonText,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LineItemId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LineItemId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersReturnLineItemRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersReturnLineItemRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersReturnLineItemResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersReturnLineItemResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersReturnLineItemResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersReturnLineItemResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersShipLineItemsRequest struct {
	// Carrier: The carrier handling the shipment.
	Carrier string `json:"carrier,omitempty"`

	// LineItems: Line items to ship.
	LineItems []*OrderShipmentLineItemShipment `json:"lineItems,omitempty"`

	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// ShipmentId: The ID of the shipment.
	ShipmentId string `json:"shipmentId,omitempty"`

	// TrackingId: The tracking id for the shipment.
	TrackingId string `json:"trackingId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carrier") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersShipLineItemsRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersShipLineItemsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersShipLineItemsResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersShipLineItemsResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersShipLineItemsResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersShipLineItemsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersUpdateMerchantOrderIdRequest struct {
	// MerchantOrderId: The merchant order id to be assigned to the order.
	// Must be unique per merchant.
	MerchantOrderId string `json:"merchantOrderId,omitempty"`

	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MerchantOrderId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MerchantOrderId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersUpdateMerchantOrderIdRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateMerchantOrderIdRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersUpdateMerchantOrderIdResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersUpdateMerchantOrderIdResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersUpdateMerchantOrderIdResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateMerchantOrderIdResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersUpdateShipmentRequest struct {
	// Carrier: The carrier handling the shipment. Not updated if missing.
	Carrier string `json:"carrier,omitempty"`

	// OperationId: The ID of the operation. Unique across all operations
	// for a given order.
	OperationId string `json:"operationId,omitempty"`

	// ShipmentId: The ID of the shipment.
	ShipmentId string `json:"shipmentId,omitempty"`

	// Status: New status for the shipment. Not updated if missing.
	Status string `json:"status,omitempty"`

	// TrackingId: The tracking id for the shipment. Not updated if missing.
	TrackingId string `json:"trackingId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carrier") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrdersUpdateShipmentRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateShipmentRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OrdersUpdateShipmentResponse struct {
	// ExecutionStatus: The status of the execution.
	ExecutionStatus string `json:"executionStatus,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#ordersUpdateShipmentResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ExecutionStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutionStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrdersUpdateShipmentResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateShipmentResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type PostalCodeGroup struct {
	// Country: The CLDR territory code of the country the postal code group
	// applies to. Required.
	Country string `json:"country,omitempty"`

	// Name: The name of the postal code group, referred to in headers.
	// Required.
	Name string `json:"name,omitempty"`

	// PostalCodeRanges: A range of postal codes. Required.
	PostalCodeRanges []*PostalCodeRange `json:"postalCodeRanges,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PostalCodeGroup) MarshalJSON() ([]byte, error) {
	type noMethod PostalCodeGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type PostalCodeRange struct {
	// PostalCodeRangeBegin: A postal code or a pattern of the form prefix*
	// denoting the inclusive lower bound of the range defining the area.
	// Examples values: "94108", "9410*", "9*". Required.
	PostalCodeRangeBegin string `json:"postalCodeRangeBegin,omitempty"`

	// PostalCodeRangeEnd: A postal code or a pattern of the form prefix*
	// denoting the inclusive upper bound of the range defining the area. It
	// must have the same length as postalCodeRangeBegin: if
	// postalCodeRangeBegin is a postal code then postalCodeRangeEnd must be
	// a postal code too; if postalCodeRangeBegin is a pattern then
	// postalCodeRangeEnd must be a pattern with the same prefix length.
	// Optional: if not set, then the area is defined as being all the
	// postal codes matching postalCodeRangeBegin.
	PostalCodeRangeEnd string `json:"postalCodeRangeEnd,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "PostalCodeRangeBegin") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PostalCodeRangeBegin") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *PostalCodeRange) MarshalJSON() ([]byte, error) {
	type noMethod PostalCodeRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Price struct {
	// Currency: The currency of the price.
	Currency string `json:"currency,omitempty"`

	// Value: The price represented as a number.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Currency") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Currency") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Price) MarshalJSON() ([]byte, error) {
	type noMethod Price
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Product: Product data.
type Product struct {
	// AdditionalImageLinks: Additional URLs of images of the item.
	AdditionalImageLinks []string `json:"additionalImageLinks,omitempty"`

	// AdditionalProductTypes: Additional categories of the item (formatted
	// as in products feed specification).
	AdditionalProductTypes []string `json:"additionalProductTypes,omitempty"`

	// Adult: Set to true if the item is targeted towards adults.
	Adult bool `json:"adult,omitempty"`

	// AdwordsGrouping: Used to group items in an arbitrary way. Only for
	// CPA%, discouraged otherwise.
	AdwordsGrouping string `json:"adwordsGrouping,omitempty"`

	// AdwordsLabels: Similar to adwords_grouping, but only works on CPC.
	AdwordsLabels []string `json:"adwordsLabels,omitempty"`

	// AdwordsRedirect: Allows advertisers to override the item URL when the
	// product is shown within the context of Product Ads.
	AdwordsRedirect string `json:"adwordsRedirect,omitempty"`

	// AgeGroup: Target age group of the item.
	AgeGroup string `json:"ageGroup,omitempty"`

	// Aspects: Specifies the intended aspects for the product.
	Aspects []*ProductAspect `json:"aspects,omitempty"`

	// Availability: Availability status of the item.
	Availability string `json:"availability,omitempty"`

	// AvailabilityDate: The day a pre-ordered product becomes available for
	// delivery, in ISO 8601 format.
	AvailabilityDate string `json:"availabilityDate,omitempty"`

	// Brand: Brand of the item.
	Brand string `json:"brand,omitempty"`

	// Channel: The item's channel (online or local).
	Channel string `json:"channel,omitempty"`

	// Color: Color of the item.
	Color string `json:"color,omitempty"`

	// Condition: Condition or state of the item.
	Condition string `json:"condition,omitempty"`

	// ContentLanguage: The two-letter ISO 639-1 language code for the item.
	ContentLanguage string `json:"contentLanguage,omitempty"`

	// CustomAttributes: A list of custom (merchant-provided) attributes. It
	// can also be used for submitting any attribute of the feed
	// specification in its generic form (e.g., { "name": "size type",
	// "type": "text", "value": "regular" }). This is useful for submitting
	// attributes not explicitly exposed by the API.
	CustomAttributes []*ProductCustomAttribute `json:"customAttributes,omitempty"`

	// CustomGroups: A list of custom (merchant-provided) custom attribute
	// groups.
	CustomGroups []*ProductCustomGroup `json:"customGroups,omitempty"`

	// CustomLabel0: Custom label 0 for custom grouping of items in a
	// Shopping campaign.
	CustomLabel0 string `json:"customLabel0,omitempty"`

	// CustomLabel1: Custom label 1 for custom grouping of items in a
	// Shopping campaign.
	CustomLabel1 string `json:"customLabel1,omitempty"`

	// CustomLabel2: Custom label 2 for custom grouping of items in a
	// Shopping campaign.
	CustomLabel2 string `json:"customLabel2,omitempty"`

	// CustomLabel3: Custom label 3 for custom grouping of items in a
	// Shopping campaign.
	CustomLabel3 string `json:"customLabel3,omitempty"`

	// CustomLabel4: Custom label 4 for custom grouping of items in a
	// Shopping campaign.
	CustomLabel4 string `json:"customLabel4,omitempty"`

	// Description: Description of the item.
	Description string `json:"description,omitempty"`

	// Destinations: Specifies the intended destinations for the product.
	Destinations []*ProductDestination `json:"destinations,omitempty"`

	// DisplayAdsId: An identifier for an item for dynamic remarketing
	// campaigns.
	DisplayAdsId string `json:"displayAdsId,omitempty"`

	// DisplayAdsLink: URL directly to your item's landing page for dynamic
	// remarketing campaigns.
	DisplayAdsLink string `json:"displayAdsLink,omitempty"`

	// DisplayAdsSimilarIds: Advertiser-specified recommendations.
	DisplayAdsSimilarIds []string `json:"displayAdsSimilarIds,omitempty"`

	// DisplayAdsTitle: Title of an item for dynamic remarketing campaigns.
	DisplayAdsTitle string `json:"displayAdsTitle,omitempty"`

	// DisplayAdsValue: Offer margin for dynamic remarketing campaigns.
	DisplayAdsValue float64 `json:"displayAdsValue,omitempty"`

	// EnergyEfficiencyClass: The energy efficiency class as defined in EU
	// directive 2010/30/EU.
	EnergyEfficiencyClass string `json:"energyEfficiencyClass,omitempty"`

	// ExpirationDate: Date on which the item should expire, as specified
	// upon insertion, in ISO 8601 format. The actual expiration date in
	// Google Shopping is exposed in productstatuses as googleExpirationDate
	// and might be earlier if expirationDate is too far in the future.
	ExpirationDate string `json:"expirationDate,omitempty"`

	// Gender: Target gender of the item.
	Gender string `json:"gender,omitempty"`

	// GoogleProductCategory: Google's category of the item (see Google
	// product taxonomy).
	GoogleProductCategory string `json:"googleProductCategory,omitempty"`

	// Gtin: Global Trade Item Number (GTIN) of the item.
	Gtin string `json:"gtin,omitempty"`

	// Id: The REST id of the product.
	Id string `json:"id,omitempty"`

	// IdentifierExists: False when the item does not have unique product
	// identifiers appropriate to its category, such as GTIN, MPN, and
	// brand. Required according to the Unique Product Identifier Rules for
	// all target countries except for Canada.
	IdentifierExists bool `json:"identifierExists,omitempty"`

	// ImageLink: URL of an image of the item.
	ImageLink string `json:"imageLink,omitempty"`

	// Installment: Number and amount of installments to pay for an item.
	// Brazil only.
	Installment *Installment `json:"installment,omitempty"`

	// IsBundle: Whether the item is a merchant-defined bundle. A bundle is
	// a custom grouping of different products sold by a merchant for a
	// single price.
	IsBundle bool `json:"isBundle,omitempty"`

	// ItemGroupId: Shared identifier for all variants of the same product.
	ItemGroupId string `json:"itemGroupId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#product".
	Kind string `json:"kind,omitempty"`

	// Link: URL directly linking to your item's page on your website.
	Link string `json:"link,omitempty"`

	// LoyaltyPoints: Loyalty points that users receive after purchasing the
	// item. Japan only.
	LoyaltyPoints *LoyaltyPoints `json:"loyaltyPoints,omitempty"`

	// Material: The material of which the item is made.
	Material string `json:"material,omitempty"`

	// MaxHandlingTime: Maximal product handling time (in business days).
	MaxHandlingTime int64 `json:"maxHandlingTime,omitempty,string"`

	// MinHandlingTime: Minimal product handling time (in business days).
	MinHandlingTime int64 `json:"minHandlingTime,omitempty,string"`

	// MobileLink: Link to a mobile-optimized version of the landing page.
	MobileLink string `json:"mobileLink,omitempty"`

	// Mpn: Manufacturer Part Number (MPN) of the item.
	Mpn string `json:"mpn,omitempty"`

	// Multipack: The number of identical products in a merchant-defined
	// multipack.
	Multipack int64 `json:"multipack,omitempty,string"`

	// OfferId: An identifier of the item. Leading and trailing whitespaces
	// are stripped and multiple whitespaces are replaced by a single
	// whitespace upon submission. Only valid unicode characters are
	// accepted. See the products feed specification for details.
	OfferId string `json:"offerId,omitempty"`

	// OnlineOnly: Whether an item is available for purchase only online.
	OnlineOnly bool `json:"onlineOnly,omitempty"`

	// Pattern: The item's pattern (e.g. polka dots).
	Pattern string `json:"pattern,omitempty"`

	// Price: Price of the item.
	Price *Price `json:"price,omitempty"`

	// ProductType: Your category of the item (formatted as in products feed
	// specification).
	ProductType string `json:"productType,omitempty"`

	// PromotionIds: The unique ID of a promotion.
	PromotionIds []string `json:"promotionIds,omitempty"`

	// SalePrice: Advertised sale price of the item.
	SalePrice *Price `json:"salePrice,omitempty"`

	// SalePriceEffectiveDate: Date range during which the item is on sale
	// (see products feed specification).
	SalePriceEffectiveDate string `json:"salePriceEffectiveDate,omitempty"`

	// SellOnGoogleQuantity: The quantity of the product that is reserved
	// for sell-on-google ads.
	SellOnGoogleQuantity int64 `json:"sellOnGoogleQuantity,omitempty,string"`

	// Shipping: Shipping rules.
	Shipping []*ProductShipping `json:"shipping,omitempty"`

	// ShippingHeight: Height of the item for shipping.
	ShippingHeight *ProductShippingDimension `json:"shippingHeight,omitempty"`

	// ShippingLabel: The shipping label of the product, used to group
	// product in account-level shipping rules.
	ShippingLabel string `json:"shippingLabel,omitempty"`

	// ShippingLength: Length of the item for shipping.
	ShippingLength *ProductShippingDimension `json:"shippingLength,omitempty"`

	// ShippingWeight: Weight of the item for shipping.
	ShippingWeight *ProductShippingWeight `json:"shippingWeight,omitempty"`

	// ShippingWidth: Width of the item for shipping.
	ShippingWidth *ProductShippingDimension `json:"shippingWidth,omitempty"`

	// SizeSystem: System in which the size is specified. Recommended for
	// apparel items.
	SizeSystem string `json:"sizeSystem,omitempty"`

	// SizeType: The cut of the item. Recommended for apparel items.
	SizeType string `json:"sizeType,omitempty"`

	// Sizes: Size of the item.
	Sizes []string `json:"sizes,omitempty"`

	// TargetCountry: The CLDR territory code for the item.
	TargetCountry string `json:"targetCountry,omitempty"`

	// Taxes: Tax information.
	Taxes []*ProductTax `json:"taxes,omitempty"`

	// Title: Title of the item.
	Title string `json:"title,omitempty"`

	// UnitPricingBaseMeasure: The preference of the denominator of the unit
	// price.
	UnitPricingBaseMeasure *ProductUnitPricingBaseMeasure `json:"unitPricingBaseMeasure,omitempty"`

	// UnitPricingMeasure: The measure and dimension of an item.
	UnitPricingMeasure *ProductUnitPricingMeasure `json:"unitPricingMeasure,omitempty"`

	// ValidatedDestinations: The read-only list of intended destinations
	// which passed validation.
	ValidatedDestinations []string `json:"validatedDestinations,omitempty"`

	// Warnings: Read-only warnings.
	Warnings []*Error `json:"warnings,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "AdditionalImageLinks") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdditionalImageLinks") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Product) MarshalJSON() ([]byte, error) {
	type noMethod Product
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Product) UnmarshalJSON(data []byte) error {
	type noMethod Product
	var s1 struct {
		DisplayAdsValue gensupport.JSONFloat64 `json:"displayAdsValue"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.DisplayAdsValue = float64(s1.DisplayAdsValue)
	return nil
}

type ProductAspect struct {
	// AspectName: The name of the aspect.
	AspectName string `json:"aspectName,omitempty"`

	// DestinationName: The name of the destination. Leave out to apply to
	// all destinations.
	DestinationName string `json:"destinationName,omitempty"`

	// Intention: Whether the aspect is required, excluded or should be
	// validated.
	Intention string `json:"intention,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AspectName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AspectName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductAspect) MarshalJSON() ([]byte, error) {
	type noMethod ProductAspect
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductCustomAttribute struct {
	// Name: The name of the attribute. Underscores will be replaced by
	// spaces upon insertion.
	Name string `json:"name,omitempty"`

	// Type: The type of the attribute.
	Type string `json:"type,omitempty"`

	// Unit: Free-form unit of the attribute. Unit can only be used for
	// values of type INT or FLOAT.
	Unit string `json:"unit,omitempty"`

	// Value: The value of the attribute.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Name") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductCustomAttribute) MarshalJSON() ([]byte, error) {
	type noMethod ProductCustomAttribute
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductCustomGroup struct {
	// Attributes: The sub-attributes.
	Attributes []*ProductCustomAttribute `json:"attributes,omitempty"`

	// Name: The name of the group. Underscores will be replaced by spaces
	// upon insertion.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Attributes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Attributes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductCustomGroup) MarshalJSON() ([]byte, error) {
	type noMethod ProductCustomGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductDestination struct {
	// DestinationName: The name of the destination.
	DestinationName string `json:"destinationName,omitempty"`

	// Intention: Whether the destination is required, excluded or should be
	// validated.
	Intention string `json:"intention,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DestinationName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DestinationName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ProductDestination) MarshalJSON() ([]byte, error) {
	type noMethod ProductDestination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductShipping struct {
	// Country: The CLDR territory code of the country to which an item will
	// ship.
	Country string `json:"country,omitempty"`

	// LocationGroupName: The location where the shipping is applicable,
	// represented by a location group name.
	LocationGroupName string `json:"locationGroupName,omitempty"`

	// LocationId: The numeric id of a location that the shipping rate
	// applies to as defined in the AdWords API.
	LocationId int64 `json:"locationId,omitempty,string"`

	// PostalCode: The postal code range that the shipping rate applies to,
	// represented by a postal code, a postal code prefix followed by a *
	// wildcard, a range between two postal codes or two postal code
	// prefixes of equal length.
	PostalCode string `json:"postalCode,omitempty"`

	// Price: Fixed shipping price, represented as a number.
	Price *Price `json:"price,omitempty"`

	// Region: The geographic region to which a shipping rate applies.
	Region string `json:"region,omitempty"`

	// Service: A free-form description of the service class or delivery
	// speed.
	Service string `json:"service,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductShipping) MarshalJSON() ([]byte, error) {
	type noMethod ProductShipping
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductShippingDimension struct {
	// Unit: The unit of value.
	//
	// Acceptable values are:
	// - "cm"
	// - "in"
	Unit string `json:"unit,omitempty"`

	// Value: The dimension of the product used to calculate the shipping
	// cost of the item.
	Value float64 `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Unit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Unit") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductShippingDimension) MarshalJSON() ([]byte, error) {
	type noMethod ProductShippingDimension
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ProductShippingDimension) UnmarshalJSON(data []byte) error {
	type noMethod ProductShippingDimension
	var s1 struct {
		Value gensupport.JSONFloat64 `json:"value"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Value = float64(s1.Value)
	return nil
}

type ProductShippingWeight struct {
	// Unit: The unit of value.
	Unit string `json:"unit,omitempty"`

	// Value: The weight of the product used to calculate the shipping cost
	// of the item.
	Value float64 `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Unit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Unit") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductShippingWeight) MarshalJSON() ([]byte, error) {
	type noMethod ProductShippingWeight
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ProductShippingWeight) UnmarshalJSON(data []byte) error {
	type noMethod ProductShippingWeight
	var s1 struct {
		Value gensupport.JSONFloat64 `json:"value"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Value = float64(s1.Value)
	return nil
}

// ProductStatus: The status of a product, i.e., information about a
// product computed asynchronously by the data quality analysis.
type ProductStatus struct {
	// CreationDate: Date on which the item has been created, in ISO 8601
	// format.
	CreationDate string `json:"creationDate,omitempty"`

	// DataQualityIssues: A list of data quality issues associated with the
	// product.
	DataQualityIssues []*ProductStatusDataQualityIssue `json:"dataQualityIssues,omitempty"`

	// DestinationStatuses: The intended destinations for the product.
	DestinationStatuses []*ProductStatusDestinationStatus `json:"destinationStatuses,omitempty"`

	// GoogleExpirationDate: Date on which the item expires in Google
	// Shopping, in ISO 8601 format.
	GoogleExpirationDate string `json:"googleExpirationDate,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#productStatus".
	Kind string `json:"kind,omitempty"`

	// LastUpdateDate: Date on which the item has been last updated, in ISO
	// 8601 format.
	LastUpdateDate string `json:"lastUpdateDate,omitempty"`

	// Link: The link to the product.
	Link string `json:"link,omitempty"`

	// Product: Product data after applying all the join inputs.
	Product *Product `json:"product,omitempty"`

	// ProductId: The id of the product for which status is reported.
	ProductId string `json:"productId,omitempty"`

	// Title: The title of the product.
	Title string `json:"title,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationDate") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductStatus) MarshalJSON() ([]byte, error) {
	type noMethod ProductStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductStatusDataQualityIssue struct {
	// Detail: A more detailed error string.
	Detail string `json:"detail,omitempty"`

	// FetchStatus: The fetch status for landing_page_errors.
	FetchStatus string `json:"fetchStatus,omitempty"`

	// Id: The id of the data quality issue.
	Id string `json:"id,omitempty"`

	// Location: The attribute name that is relevant for the issue.
	Location string `json:"location,omitempty"`

	// Severity: The severity of the data quality issue.
	Severity string `json:"severity,omitempty"`

	// Timestamp: The time stamp of the data quality issue.
	Timestamp string `json:"timestamp,omitempty"`

	// ValueOnLandingPage: The value of that attribute that was found on the
	// landing page
	ValueOnLandingPage string `json:"valueOnLandingPage,omitempty"`

	// ValueProvided: The value the attribute had at time of evaluation.
	ValueProvided string `json:"valueProvided,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Detail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Detail") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductStatusDataQualityIssue) MarshalJSON() ([]byte, error) {
	type noMethod ProductStatusDataQualityIssue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductStatusDestinationStatus struct {
	// ApprovalStatus: The destination's approval status.
	ApprovalStatus string `json:"approvalStatus,omitempty"`

	// Destination: The name of the destination
	Destination string `json:"destination,omitempty"`

	// Intention: Whether the destination is required, excluded, selected by
	// default or should be validated.
	Intention string `json:"intention,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApprovalStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApprovalStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ProductStatusDestinationStatus) MarshalJSON() ([]byte, error) {
	type noMethod ProductStatusDestinationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductTax struct {
	// Country: The country within which the item is taxed, specified as a
	// CLDR territory code.
	Country string `json:"country,omitempty"`

	// LocationId: The numeric id of a location that the tax rate applies to
	// as defined in the AdWords API.
	LocationId int64 `json:"locationId,omitempty,string"`

	// PostalCode: The postal code range that the tax rate applies to,
	// represented by a ZIP code, a ZIP code prefix using * wildcard, a
	// range between two ZIP codes or two ZIP code prefixes of equal length.
	// Examples: 94114, 94*, 94002-95460, 94*-95*.
	PostalCode string `json:"postalCode,omitempty"`

	// Rate: The percentage of tax rate that applies to the item price.
	Rate float64 `json:"rate,omitempty"`

	// Region: The geographic region to which the tax rate applies.
	Region string `json:"region,omitempty"`

	// TaxShip: Set to true if tax is charged on shipping.
	TaxShip bool `json:"taxShip,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductTax) MarshalJSON() ([]byte, error) {
	type noMethod ProductTax
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ProductTax) UnmarshalJSON(data []byte) error {
	type noMethod ProductTax
	var s1 struct {
		Rate gensupport.JSONFloat64 `json:"rate"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Rate = float64(s1.Rate)
	return nil
}

type ProductUnitPricingBaseMeasure struct {
	// Unit: The unit of the denominator.
	Unit string `json:"unit,omitempty"`

	// Value: The denominator of the unit price.
	Value int64 `json:"value,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Unit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Unit") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductUnitPricingBaseMeasure) MarshalJSON() ([]byte, error) {
	type noMethod ProductUnitPricingBaseMeasure
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductUnitPricingMeasure struct {
	// Unit: The unit of the measure.
	Unit string `json:"unit,omitempty"`

	// Value: The measure of an item.
	Value float64 `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Unit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Unit") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductUnitPricingMeasure) MarshalJSON() ([]byte, error) {
	type noMethod ProductUnitPricingMeasure
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ProductUnitPricingMeasure) UnmarshalJSON(data []byte) error {
	type noMethod ProductUnitPricingMeasure
	var s1 struct {
		Value gensupport.JSONFloat64 `json:"value"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Value = float64(s1.Value)
	return nil
}

type ProductsCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*ProductsCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductsCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductsCustomBatchRequestEntry: A batch entry encoding a single
// non-batch products request.
type ProductsCustomBatchRequestEntry struct {
	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// Product: The product to insert. Only required if the method is
	// insert.
	Product *Product `json:"product,omitempty"`

	// ProductId: The ID of the product to get or delete. Only defined if
	// the method is get or delete.
	ProductId string `json:"productId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductsCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductsCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*ProductsCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#productsCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductsCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductsCustomBatchResponseEntry: A batch entry encoding a single
// non-batch products response.
type ProductsCustomBatchResponseEntry struct {
	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#productsCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// Product: The inserted product. Only defined if the method is insert
	// and if the request was successful.
	Product *Product `json:"product,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductsCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductsListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#productsListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// products.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*Product `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductstatusesCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*ProductstatusesCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductstatusesCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductstatusesCustomBatchRequestEntry: A batch entry encoding a
// single non-batch productstatuses request.
type ProductstatusesCustomBatchRequestEntry struct {
	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	IncludeAttributes bool `json:"includeAttributes,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// ProductId: The ID of the product whose status to get.
	ProductId string `json:"productId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductstatusesCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductstatusesCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*ProductstatusesCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#productstatusesCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductstatusesCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductstatusesCustomBatchResponseEntry: A batch entry encoding a
// single non-batch productstatuses response.
type ProductstatusesCustomBatchResponseEntry struct {
	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors, if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#productstatusesCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// ProductStatus: The requested product status. Only defined if the
	// request was successful.
	ProductStatus *ProductStatus `json:"productStatus,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductstatusesCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductstatusesListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#productstatusesListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// products statuses.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*ProductStatus `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductstatusesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type RateGroup struct {
	// ApplicableShippingLabels: A list of shipping labels defining the
	// products to which this rate group applies to. This is a disjunction:
	// only one of the labels has to match for the rate group to apply. May
	// only be empty for the last rate group of a service. Required.
	ApplicableShippingLabels []string `json:"applicableShippingLabels,omitempty"`

	// CarrierRates: A list of carrier rates that can be referred to by
	// mainTable or singleValue.
	CarrierRates []*CarrierRate `json:"carrierRates,omitempty"`

	// MainTable: A table defining the rate group, when singleValue is not
	// expressive enough. Can only be set if singleValue is not set.
	MainTable *Table `json:"mainTable,omitempty"`

	// SingleValue: The value of the rate group (e.g. flat rate $10). Can
	// only be set if mainTable and subtables are not set.
	SingleValue *Value `json:"singleValue,omitempty"`

	// Subtables: A list of subtables referred to by mainTable. Can only be
	// set if mainTable is set.
	Subtables []*Table `json:"subtables,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ApplicableShippingLabels") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApplicableShippingLabels")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *RateGroup) MarshalJSON() ([]byte, error) {
	type noMethod RateGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Row struct {
	// Cells: The list of cells that constitute the row. Must have the same
	// length as columnHeaders for two-dimensional tables, a length of 1 for
	// one-dimensional tables. Required.
	Cells []*Value `json:"cells,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cells") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cells") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Row) MarshalJSON() ([]byte, error) {
	type noMethod Row
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Service struct {
	// Active: A boolean exposing the active status of the shipping service.
	// Required.
	Active bool `json:"active,omitempty"`

	// Currency: The CLDR code of the currency to which this service
	// applies. Must match that of the prices in rate groups.
	Currency string `json:"currency,omitempty"`

	// DeliveryCountry: The CLDR territory code of the country to which the
	// service applies. Required.
	DeliveryCountry string `json:"deliveryCountry,omitempty"`

	// DeliveryTime: Time spent in various aspects from order to the
	// delivery of the product. Required.
	DeliveryTime *DeliveryTime `json:"deliveryTime,omitempty"`

	// MinimumOrderValue: Minimum order value for this service. If set,
	// indicates that customers will have to spend at least this amount. All
	// prices within a service must have the same currency.
	MinimumOrderValue *Price `json:"minimumOrderValue,omitempty"`

	// Name: Free-form name of the service. Must be unique within target
	// account. Required.
	Name string `json:"name,omitempty"`

	// RateGroups: Shipping rate group definitions. Only the last one is
	// allowed to have an empty applicableShippingLabels, which means
	// "everything else". The other applicableShippingLabels must not
	// overlap.
	RateGroups []*RateGroup `json:"rateGroups,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Active") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Active") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Service) MarshalJSON() ([]byte, error) {
	type noMethod Service
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ShippingSettings: The merchant account's shipping settings.
type ShippingSettings struct {
	// AccountId: The ID of the account to which these account shipping
	// settings belong. Ignored upon update, always present in get request
	// responses.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// PostalCodeGroups: A list of postal code groups that can be referred
	// to in services. Optional.
	PostalCodeGroups []*PostalCodeGroup `json:"postalCodeGroups,omitempty"`

	// Services: The target account's list of services. Optional.
	Services []*Service `json:"services,omitempty"`

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

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ShippingSettings) MarshalJSON() ([]byte, error) {
	type noMethod ShippingSettings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ShippingsettingsCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*ShippingsettingsCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ShippingsettingsCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod ShippingsettingsCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ShippingsettingsCustomBatchRequestEntry: A batch entry encoding a
// single non-batch shippingsettings request.
type ShippingsettingsCustomBatchRequestEntry struct {
	// AccountId: The ID of the account for which to get/update account
	// shipping settings.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// ShippingSettings: The account shipping settings to update. Only
	// defined if the method is update.
	ShippingSettings *ShippingSettings `json:"shippingSettings,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ShippingsettingsCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod ShippingsettingsCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ShippingsettingsCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*ShippingsettingsCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#shippingsettingsCustomBatchResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ShippingsettingsCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod ShippingsettingsCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ShippingsettingsCustomBatchResponseEntry: A batch entry encoding a
// single non-batch shipping settings response.
type ShippingsettingsCustomBatchResponseEntry struct {
	// BatchId: The ID of the request entry to which this entry responds.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if, and only if, the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#shippingsettingsCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// ShippingSettings: The retrieved or updated account shipping settings.
	ShippingSettings *ShippingSettings `json:"shippingSettings,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatchId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatchId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ShippingsettingsCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod ShippingsettingsCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ShippingsettingsGetSupportedCarriersResponse struct {
	// Carriers: A list of supported carriers. May be empty.
	Carriers []*CarriersCarrier `json:"carriers,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#shippingsettingsGetSupportedCarriersResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Carriers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carriers") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ShippingsettingsGetSupportedCarriersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ShippingsettingsGetSupportedCarriersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ShippingsettingsListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#shippingsettingsListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// shipping settings.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*ShippingSettings `json:"resources,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ShippingsettingsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ShippingsettingsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Table struct {
	// ColumnHeaders: Headers of the table's columns. Optional: if not set
	// then the table has only one dimension.
	ColumnHeaders *Headers `json:"columnHeaders,omitempty"`

	// Name: Name of the table. Required for subtables, ignored for the main
	// table.
	Name string `json:"name,omitempty"`

	// RowHeaders: Headers of the table's rows. Required.
	RowHeaders *Headers `json:"rowHeaders,omitempty"`

	// Rows: The list of rows that constitute the table. Must have the same
	// length as rowHeaders. Required.
	Rows []*Row `json:"rows,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnHeaders") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnHeaders") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Table) MarshalJSON() ([]byte, error) {
	type noMethod Table
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TestOrder struct {
	// Customer: The details of the customer who placed the order.
	Customer *TestOrderCustomer `json:"customer,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#testOrder".
	Kind string `json:"kind,omitempty"`

	// LineItems: Line items that are ordered. At least one line item must
	// be provided.
	LineItems []*TestOrderLineItem `json:"lineItems,omitempty"`

	// PaymentMethod: The details of the payment method.
	PaymentMethod *TestOrderPaymentMethod `json:"paymentMethod,omitempty"`

	// PredefinedDeliveryAddress: Identifier of one of the predefined
	// delivery addresses for the delivery.
	PredefinedDeliveryAddress string `json:"predefinedDeliveryAddress,omitempty"`

	// Promotions: The details of the merchant provided promotions applied
	// to the order. More details about the program are here.
	Promotions []*OrderPromotion `json:"promotions,omitempty"`

	// ShippingCost: The total cost of shipping for all items.
	ShippingCost *Price `json:"shippingCost,omitempty"`

	// ShippingCostTax: The tax for the total shipping cost.
	ShippingCostTax *Price `json:"shippingCostTax,omitempty"`

	// ShippingOption: The requested shipping option.
	ShippingOption string `json:"shippingOption,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Customer") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Customer") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestOrder) MarshalJSON() ([]byte, error) {
	type noMethod TestOrder
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TestOrderCustomer struct {
	// Email: Email address of the customer.
	Email string `json:"email,omitempty"`

	// ExplicitMarketingPreference: If set, this indicates the user
	// explicitly chose to opt in or out of providing marketing rights to
	// the merchant. If unset, this indicates the user has already made this
	// choice in a previous purchase, and was thus not shown the marketing
	// right opt in/out checkbox during the checkout flow. Optional.
	ExplicitMarketingPreference bool `json:"explicitMarketingPreference,omitempty"`

	// FullName: Full name of the customer.
	FullName string `json:"fullName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Email") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Email") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestOrderCustomer) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderCustomer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TestOrderLineItem struct {
	// Product: Product data from the time of the order placement.
	Product *TestOrderLineItemProduct `json:"product,omitempty"`

	// QuantityOrdered: Number of items ordered.
	QuantityOrdered int64 `json:"quantityOrdered,omitempty"`

	// ReturnInfo: Details of the return policy for the line item.
	ReturnInfo *OrderLineItemReturnInfo `json:"returnInfo,omitempty"`

	// ShippingDetails: Details of the requested shipping for the line item.
	ShippingDetails *OrderLineItemShippingDetails `json:"shippingDetails,omitempty"`

	// UnitTax: Unit tax for the line item.
	UnitTax *Price `json:"unitTax,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Product") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Product") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestOrderLineItem) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TestOrderLineItemProduct struct {
	// Brand: Brand of the item.
	Brand string `json:"brand,omitempty"`

	// Channel: The item's channel.
	Channel string `json:"channel,omitempty"`

	// Condition: Condition or state of the item.
	Condition string `json:"condition,omitempty"`

	// ContentLanguage: The two-letter ISO 639-1 language code for the item.
	ContentLanguage string `json:"contentLanguage,omitempty"`

	// Gtin: Global Trade Item Number (GTIN) of the item. Optional.
	Gtin string `json:"gtin,omitempty"`

	// ImageLink: URL of an image of the item.
	ImageLink string `json:"imageLink,omitempty"`

	// ItemGroupId: Shared identifier for all variants of the same product.
	// Optional.
	ItemGroupId string `json:"itemGroupId,omitempty"`

	// Mpn: Manufacturer Part Number (MPN) of the item. Optional.
	Mpn string `json:"mpn,omitempty"`

	// OfferId: An identifier of the item.
	OfferId string `json:"offerId,omitempty"`

	// Price: The price for the product.
	Price *Price `json:"price,omitempty"`

	// TargetCountry: The CLDR territory code of the target country of the
	// product.
	TargetCountry string `json:"targetCountry,omitempty"`

	// Title: The title of the product.
	Title string `json:"title,omitempty"`

	// VariantAttributes: Variant attributes for the item. Optional.
	VariantAttributes []*OrderLineItemProductVariantAttribute `json:"variantAttributes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Brand") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Brand") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestOrderLineItemProduct) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderLineItemProduct
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TestOrderPaymentMethod struct {
	// ExpirationMonth: The card expiration month (January = 1, February = 2
	// etc.).
	ExpirationMonth int64 `json:"expirationMonth,omitempty"`

	// ExpirationYear: The card expiration year (4-digit, e.g. 2015).
	ExpirationYear int64 `json:"expirationYear,omitempty"`

	// LastFourDigits: The last four digits of the card number.
	LastFourDigits string `json:"lastFourDigits,omitempty"`

	// PredefinedBillingAddress: The billing address.
	PredefinedBillingAddress string `json:"predefinedBillingAddress,omitempty"`

	// Type: The type of instrument. Note that real orders might have
	// different values than the four values accepted by createTestOrder.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExpirationMonth") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExpirationMonth") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TestOrderPaymentMethod) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderPaymentMethod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Value: The single value of a rate group or the value of a rate group
// table's cell. Exactly one of noShipping, flatRate, pricePercentage,
// carrierRateName, subtableName must be set.
type Value struct {
	// CarrierRateName: The name of a carrier rate referring to a carrier
	// rate defined in the same rate group. Can only be set if all other
	// fields are not set.
	CarrierRateName string `json:"carrierRateName,omitempty"`

	// FlatRate: A flat rate. Can only be set if all other fields are not
	// set.
	FlatRate *Price `json:"flatRate,omitempty"`

	// NoShipping: If true, then the product can't ship. Must be true when
	// set, can only be set if all other fields are not set.
	NoShipping bool `json:"noShipping,omitempty"`

	// PricePercentage: A percentage of the price represented as a number in
	// decimal notation (e.g., "5.4"). Can only be set if all other fields
	// are not set.
	PricePercentage string `json:"pricePercentage,omitempty"`

	// SubtableName: The name of a subtable. Can only be set in table cells
	// (i.e., not for single values), and only if all other fields are not
	// set.
	SubtableName string `json:"subtableName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CarrierRateName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CarrierRateName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Value) MarshalJSON() ([]byte, error) {
	type noMethod Value
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Weight struct {
	// Unit: The weight unit.
	Unit string `json:"unit,omitempty"`

	// Value: The weight represented as a number.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Unit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Unit") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Weight) MarshalJSON() ([]byte, error) {
	type noMethod Weight
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "content.accounts.authinfo":

type AccountsAuthinfoCall struct {
	s            *APIService
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Authinfo: Returns information about the authenticated user.
func (r *AccountsService) Authinfo() *AccountsAuthinfoCall {
	c := &AccountsAuthinfoCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsAuthinfoCall) Fields(s ...googleapi.Field) *AccountsAuthinfoCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsAuthinfoCall) IfNoneMatch(entityTag string) *AccountsAuthinfoCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsAuthinfoCall) Context(ctx context.Context) *AccountsAuthinfoCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsAuthinfoCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsAuthinfoCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/authinfo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.authinfo" call.
// Exactly one of *AccountsAuthInfoResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AccountsAuthInfoResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsAuthinfoCall) Do(opts ...googleapi.CallOption) (*AccountsAuthInfoResponse, error) {
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
	ret := &AccountsAuthInfoResponse{
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
	//   "description": "Returns information about the authenticated user.",
	//   "httpMethod": "GET",
	//   "id": "content.accounts.authinfo",
	//   "path": "accounts/authinfo",
	//   "response": {
	//     "$ref": "AccountsAuthInfoResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounts.claimwebsite":

type AccountsClaimwebsiteCall struct {
	s          *APIService
	merchantId uint64
	accountId  uint64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Claimwebsite: Claims the website of a Merchant Center sub-account.
// This method can only be called for accounts to which the managing
// account has access: either the managing account itself for any
// Merchant Center account, or any sub-account when the managing account
// is a multi-client account.
func (r *AccountsService) Claimwebsite(merchantId uint64, accountId uint64) *AccountsClaimwebsiteCall {
	c := &AccountsClaimwebsiteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// Overwrite sets the optional parameter "overwrite": Only available to
// selected merchants. When set to True, this flag removes any existing
// claim on the requested website by another account and replaces it
// with a claim from this account.
func (c *AccountsClaimwebsiteCall) Overwrite(overwrite bool) *AccountsClaimwebsiteCall {
	c.urlParams_.Set("overwrite", fmt.Sprint(overwrite))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClaimwebsiteCall) Fields(s ...googleapi.Field) *AccountsClaimwebsiteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClaimwebsiteCall) Context(ctx context.Context) *AccountsClaimwebsiteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClaimwebsiteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClaimwebsiteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}/claimwebsite")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.claimwebsite" call.
// Exactly one of *AccountsClaimWebsiteResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *AccountsClaimWebsiteResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsClaimwebsiteCall) Do(opts ...googleapi.CallOption) (*AccountsClaimWebsiteResponse, error) {
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
	ret := &AccountsClaimWebsiteResponse{
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
	//   "description": "Claims the website of a Merchant Center sub-account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "POST",
	//   "id": "content.accounts.claimwebsite",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account whose website is claimed.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "overwrite": {
	//       "description": "Only available to selected merchants. When set to True, this flag removes any existing claim on the requested website by another account and replaces it with a claim from this account.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "{merchantId}/accounts/{accountId}/claimwebsite",
	//   "response": {
	//     "$ref": "AccountsClaimWebsiteResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounts.custombatch":

type AccountsCustombatchCall struct {
	s                          *APIService
	accountscustombatchrequest *AccountsCustomBatchRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// Custombatch: Retrieves, inserts, updates, and deletes multiple
// Merchant Center (sub-)accounts in a single request.
func (r *AccountsService) Custombatch(accountscustombatchrequest *AccountsCustomBatchRequest) *AccountsCustombatchCall {
	c := &AccountsCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountscustombatchrequest = accountscustombatchrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountsCustombatchCall) DryRun(dryRun bool) *AccountsCustombatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCustombatchCall) Fields(s ...googleapi.Field) *AccountsCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCustombatchCall) Context(ctx context.Context) *AccountsCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accountscustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.custombatch" call.
// Exactly one of *AccountsCustomBatchResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AccountsCustomBatchResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsCustombatchCall) Do(opts ...googleapi.CallOption) (*AccountsCustomBatchResponse, error) {
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
	ret := &AccountsCustomBatchResponse{
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
	//   "description": "Retrieves, inserts, updates, and deletes multiple Merchant Center (sub-)accounts in a single request.",
	//   "httpMethod": "POST",
	//   "id": "content.accounts.custombatch",
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "accounts/batch",
	//   "request": {
	//     "$ref": "AccountsCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "AccountsCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounts.delete":

type AccountsDeleteCall struct {
	s          *APIService
	merchantId uint64
	accountId  uint64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a Merchant Center sub-account. This method can only
// be called for multi-client accounts.
func (r *AccountsService) Delete(merchantId uint64, accountId uint64) *AccountsDeleteCall {
	c := &AccountsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountsDeleteCall) DryRun(dryRun bool) *AccountsDeleteCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Force sets the optional parameter "force": Flag to delete
// sub-accounts with products. The default value of false will become
// active on September 28, 2017.
func (c *AccountsDeleteCall) Force(force bool) *AccountsDeleteCall {
	c.urlParams_.Set("force", fmt.Sprint(force))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsDeleteCall) Fields(s ...googleapi.Field) *AccountsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsDeleteCall) Context(ctx context.Context) *AccountsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.delete" call.
func (c *AccountsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Deletes a Merchant Center sub-account. This method can only be called for multi-client accounts.",
	//   "httpMethod": "DELETE",
	//   "id": "content.accounts.delete",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "force": {
	//       "default": "true",
	//       "description": "Flag to delete sub-accounts with products. The default value of false will become active on September 28, 2017.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounts/{accountId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounts.get":

type AccountsGetCall struct {
	s            *APIService
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a Merchant Center account. This method can only be
// called for accounts to which the managing account has access: either
// the managing account itself for any Merchant Center account, or any
// sub-account when the managing account is a multi-client account.
func (r *AccountsService) Get(merchantId uint64, accountId uint64) *AccountsGetCall {
	c := &AccountsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
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

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.get" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsGetCall) Do(opts ...googleapi.CallOption) (*Account, error) {
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
	ret := &Account{
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
	//   "description": "Retrieves a Merchant Center account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "GET",
	//   "id": "content.accounts.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounts/{accountId}",
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounts.insert":

type AccountsInsertCall struct {
	s          *APIService
	merchantId uint64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a Merchant Center sub-account. This method can only
// be called for multi-client accounts.
func (r *AccountsService) Insert(merchantId uint64, account *Account) *AccountsInsertCall {
	c := &AccountsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.account = account
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountsInsertCall) DryRun(dryRun bool) *AccountsInsertCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsInsertCall) Fields(s ...googleapi.Field) *AccountsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsInsertCall) Context(ctx context.Context) *AccountsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.insert" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsInsertCall) Do(opts ...googleapi.CallOption) (*Account, error) {
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
	ret := &Account{
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
	//   "description": "Creates a Merchant Center sub-account. This method can only be called for multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.accounts.insert",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounts",
	//   "request": {
	//     "$ref": "Account"
	//   },
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounts.list":

type AccountsListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the sub-accounts in your Merchant Center account. This
// method can only be called for multi-client accounts.
func (r *AccountsService) List(merchantId uint64) *AccountsListCall {
	c := &AccountsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of accounts to return in the response, used for paging.
func (c *AccountsListCall) MaxResults(maxResults int64) *AccountsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *AccountsListCall) PageToken(pageToken string) *AccountsListCall {
	c.urlParams_.Set("pageToken", pageToken)
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

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.list" call.
// Exactly one of *AccountsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AccountsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsListCall) Do(opts ...googleapi.CallOption) (*AccountsListResponse, error) {
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
	ret := &AccountsListResponse{
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
	//   "description": "Lists the sub-accounts in your Merchant Center account. This method can only be called for multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.accounts.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of accounts to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounts",
	//   "response": {
	//     "$ref": "AccountsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsListCall) Pages(ctx context.Context, f func(*AccountsListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.accounts.patch":

type AccountsPatchCall struct {
	s          *APIService
	merchantId uint64
	accountId  uint64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a Merchant Center account. This method can only be
// called for accounts to which the managing account has access: either
// the managing account itself for any Merchant Center account, or any
// sub-account when the managing account is a multi-client account. This
// method supports patch semantics.
func (r *AccountsService) Patch(merchantId uint64, accountId uint64, account *Account) *AccountsPatchCall {
	c := &AccountsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.account = account
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountsPatchCall) DryRun(dryRun bool) *AccountsPatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
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

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.patch" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsPatchCall) Do(opts ...googleapi.CallOption) (*Account, error) {
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
	ret := &Account{
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
	//   "description": "Updates a Merchant Center account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "content.accounts.patch",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounts/{accountId}",
	//   "request": {
	//     "$ref": "Account"
	//   },
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounts.update":

type AccountsUpdateCall struct {
	s          *APIService
	merchantId uint64
	accountId  uint64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a Merchant Center account. This method can only be
// called for accounts to which the managing account has access: either
// the managing account itself for any Merchant Center account, or any
// sub-account when the managing account is a multi-client account.
func (r *AccountsService) Update(merchantId uint64, accountId uint64, account *Account) *AccountsUpdateCall {
	c := &AccountsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.account = account
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountsUpdateCall) DryRun(dryRun bool) *AccountsUpdateCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
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

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounts.update" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsUpdateCall) Do(opts ...googleapi.CallOption) (*Account, error) {
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
	ret := &Account{
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
	//   "description": "Updates a Merchant Center account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "PUT",
	//   "id": "content.accounts.update",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounts/{accountId}",
	//   "request": {
	//     "$ref": "Account"
	//   },
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountstatuses.custombatch":

type AccountstatusesCustombatchCall struct {
	s                                 *APIService
	accountstatusescustombatchrequest *AccountstatusesCustomBatchRequest
	urlParams_                        gensupport.URLParams
	ctx_                              context.Context
	header_                           http.Header
}

// Custombatch:
func (r *AccountstatusesService) Custombatch(accountstatusescustombatchrequest *AccountstatusesCustomBatchRequest) *AccountstatusesCustombatchCall {
	c := &AccountstatusesCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountstatusescustombatchrequest = accountstatusescustombatchrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountstatusesCustombatchCall) Fields(s ...googleapi.Field) *AccountstatusesCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountstatusesCustombatchCall) Context(ctx context.Context) *AccountstatusesCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountstatusesCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountstatusesCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accountstatusescustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accountstatuses/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accountstatuses.custombatch" call.
// Exactly one of *AccountstatusesCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *AccountstatusesCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountstatusesCustombatchCall) Do(opts ...googleapi.CallOption) (*AccountstatusesCustomBatchResponse, error) {
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
	ret := &AccountstatusesCustomBatchResponse{
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
	//   "httpMethod": "POST",
	//   "id": "content.accountstatuses.custombatch",
	//   "path": "accountstatuses/batch",
	//   "request": {
	//     "$ref": "AccountstatusesCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "AccountstatusesCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountstatuses.get":

type AccountstatusesGetCall struct {
	s            *APIService
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the status of a Merchant Center account. This method
// can only be called for accounts to which the managing account has
// access: either the managing account itself for any Merchant Center
// account, or any sub-account when the managing account is a
// multi-client account.
func (r *AccountstatusesService) Get(merchantId uint64, accountId uint64) *AccountstatusesGetCall {
	c := &AccountstatusesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountstatusesGetCall) Fields(s ...googleapi.Field) *AccountstatusesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountstatusesGetCall) IfNoneMatch(entityTag string) *AccountstatusesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountstatusesGetCall) Context(ctx context.Context) *AccountstatusesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountstatusesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountstatusesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountstatuses/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accountstatuses.get" call.
// Exactly one of *AccountStatus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountStatus.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountstatusesGetCall) Do(opts ...googleapi.CallOption) (*AccountStatus, error) {
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
	ret := &AccountStatus{
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
	//   "description": "Retrieves the status of a Merchant Center account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "GET",
	//   "id": "content.accountstatuses.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accountstatuses/{accountId}",
	//   "response": {
	//     "$ref": "AccountStatus"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountstatuses.list":

type AccountstatusesListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the statuses of the sub-accounts in your Merchant Center
// account. This method can only be called for multi-client accounts.
func (r *AccountstatusesService) List(merchantId uint64) *AccountstatusesListCall {
	c := &AccountstatusesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of account statuses to return in the response, used for
// paging.
func (c *AccountstatusesListCall) MaxResults(maxResults int64) *AccountstatusesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *AccountstatusesListCall) PageToken(pageToken string) *AccountstatusesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountstatusesListCall) Fields(s ...googleapi.Field) *AccountstatusesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountstatusesListCall) IfNoneMatch(entityTag string) *AccountstatusesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountstatusesListCall) Context(ctx context.Context) *AccountstatusesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountstatusesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountstatusesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountstatuses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accountstatuses.list" call.
// Exactly one of *AccountstatusesListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AccountstatusesListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountstatusesListCall) Do(opts ...googleapi.CallOption) (*AccountstatusesListResponse, error) {
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
	ret := &AccountstatusesListResponse{
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
	//   "description": "Lists the statuses of the sub-accounts in your Merchant Center account. This method can only be called for multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.accountstatuses.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of account statuses to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accountstatuses",
	//   "response": {
	//     "$ref": "AccountstatusesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountstatusesListCall) Pages(ctx context.Context, f func(*AccountstatusesListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.accounttax.custombatch":

type AccounttaxCustombatchCall struct {
	s                            *APIService
	accounttaxcustombatchrequest *AccounttaxCustomBatchRequest
	urlParams_                   gensupport.URLParams
	ctx_                         context.Context
	header_                      http.Header
}

// Custombatch: Retrieves and updates tax settings of multiple accounts
// in a single request.
func (r *AccounttaxService) Custombatch(accounttaxcustombatchrequest *AccounttaxCustomBatchRequest) *AccounttaxCustombatchCall {
	c := &AccounttaxCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accounttaxcustombatchrequest = accounttaxcustombatchrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccounttaxCustombatchCall) DryRun(dryRun bool) *AccounttaxCustombatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccounttaxCustombatchCall) Fields(s ...googleapi.Field) *AccounttaxCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccounttaxCustombatchCall) Context(ctx context.Context) *AccounttaxCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccounttaxCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccounttaxCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accounttaxcustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounttax/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounttax.custombatch" call.
// Exactly one of *AccounttaxCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *AccounttaxCustomBatchResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccounttaxCustombatchCall) Do(opts ...googleapi.CallOption) (*AccounttaxCustomBatchResponse, error) {
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
	ret := &AccounttaxCustomBatchResponse{
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
	//   "description": "Retrieves and updates tax settings of multiple accounts in a single request.",
	//   "httpMethod": "POST",
	//   "id": "content.accounttax.custombatch",
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "accounttax/batch",
	//   "request": {
	//     "$ref": "AccounttaxCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "AccounttaxCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounttax.get":

type AccounttaxGetCall struct {
	s            *APIService
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the tax settings of the account. This method can only
// be called for accounts to which the managing account has access:
// either the managing account itself for any Merchant Center account,
// or any sub-account when the managing account is a multi-client
// account.
func (r *AccounttaxService) Get(merchantId uint64, accountId uint64) *AccounttaxGetCall {
	c := &AccounttaxGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccounttaxGetCall) Fields(s ...googleapi.Field) *AccounttaxGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccounttaxGetCall) IfNoneMatch(entityTag string) *AccounttaxGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccounttaxGetCall) Context(ctx context.Context) *AccounttaxGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccounttaxGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccounttaxGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounttax.get" call.
// Exactly one of *AccountTax or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountTax.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccounttaxGetCall) Do(opts ...googleapi.CallOption) (*AccountTax, error) {
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
	ret := &AccountTax{
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
	//   "description": "Retrieves the tax settings of the account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "GET",
	//   "id": "content.accounttax.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update account tax settings.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounttax/{accountId}",
	//   "response": {
	//     "$ref": "AccountTax"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounttax.list":

type AccounttaxListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the tax settings of the sub-accounts in your Merchant
// Center account. This method can only be called for multi-client
// accounts.
func (r *AccounttaxService) List(merchantId uint64) *AccounttaxListCall {
	c := &AccounttaxListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of tax settings to return in the response, used for paging.
func (c *AccounttaxListCall) MaxResults(maxResults int64) *AccounttaxListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *AccounttaxListCall) PageToken(pageToken string) *AccounttaxListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccounttaxListCall) Fields(s ...googleapi.Field) *AccounttaxListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccounttaxListCall) IfNoneMatch(entityTag string) *AccounttaxListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccounttaxListCall) Context(ctx context.Context) *AccounttaxListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccounttaxListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccounttaxListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounttax.list" call.
// Exactly one of *AccounttaxListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AccounttaxListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccounttaxListCall) Do(opts ...googleapi.CallOption) (*AccounttaxListResponse, error) {
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
	ret := &AccounttaxListResponse{
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
	//   "description": "Lists the tax settings of the sub-accounts in your Merchant Center account. This method can only be called for multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.accounttax.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of tax settings to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounttax",
	//   "response": {
	//     "$ref": "AccounttaxListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccounttaxListCall) Pages(ctx context.Context, f func(*AccounttaxListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.accounttax.patch":

type AccounttaxPatchCall struct {
	s          *APIService
	merchantId uint64
	accountId  uint64
	accounttax *AccountTax
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates the tax settings of the account. This method can only
// be called for accounts to which the managing account has access:
// either the managing account itself for any Merchant Center account,
// or any sub-account when the managing account is a multi-client
// account. This method supports patch semantics.
func (r *AccounttaxService) Patch(merchantId uint64, accountId uint64, accounttax *AccountTax) *AccounttaxPatchCall {
	c := &AccounttaxPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.accounttax = accounttax
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccounttaxPatchCall) DryRun(dryRun bool) *AccounttaxPatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccounttaxPatchCall) Fields(s ...googleapi.Field) *AccounttaxPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccounttaxPatchCall) Context(ctx context.Context) *AccounttaxPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccounttaxPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccounttaxPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accounttax)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounttax.patch" call.
// Exactly one of *AccountTax or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountTax.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccounttaxPatchCall) Do(opts ...googleapi.CallOption) (*AccountTax, error) {
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
	ret := &AccountTax{
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
	//   "description": "Updates the tax settings of the account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "content.accounttax.patch",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update account tax settings.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounttax/{accountId}",
	//   "request": {
	//     "$ref": "AccountTax"
	//   },
	//   "response": {
	//     "$ref": "AccountTax"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accounttax.update":

type AccounttaxUpdateCall struct {
	s          *APIService
	merchantId uint64
	accountId  uint64
	accounttax *AccountTax
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates the tax settings of the account. This method can only
// be called for accounts to which the managing account has access:
// either the managing account itself for any Merchant Center account,
// or any sub-account when the managing account is a multi-client
// account.
func (r *AccounttaxService) Update(merchantId uint64, accountId uint64, accounttax *AccountTax) *AccounttaxUpdateCall {
	c := &AccounttaxUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.accounttax = accounttax
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccounttaxUpdateCall) DryRun(dryRun bool) *AccounttaxUpdateCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccounttaxUpdateCall) Fields(s ...googleapi.Field) *AccounttaxUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccounttaxUpdateCall) Context(ctx context.Context) *AccounttaxUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccounttaxUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccounttaxUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accounttax)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.accounttax.update" call.
// Exactly one of *AccountTax or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountTax.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccounttaxUpdateCall) Do(opts ...googleapi.CallOption) (*AccountTax, error) {
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
	ret := &AccountTax{
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
	//   "description": "Updates the tax settings of the account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "PUT",
	//   "id": "content.accounttax.update",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update account tax settings.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/accounttax/{accountId}",
	//   "request": {
	//     "$ref": "AccountTax"
	//   },
	//   "response": {
	//     "$ref": "AccountTax"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeeds.custombatch":

type DatafeedsCustombatchCall struct {
	s                           *APIService
	datafeedscustombatchrequest *DatafeedsCustomBatchRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Custombatch:
func (r *DatafeedsService) Custombatch(datafeedscustombatchrequest *DatafeedsCustomBatchRequest) *DatafeedsCustombatchCall {
	c := &DatafeedsCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datafeedscustombatchrequest = datafeedscustombatchrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *DatafeedsCustombatchCall) DryRun(dryRun bool) *DatafeedsCustombatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedsCustombatchCall) Fields(s ...googleapi.Field) *DatafeedsCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedsCustombatchCall) Context(ctx context.Context) *DatafeedsCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedsCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedsCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeedscustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "datafeeds/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeeds.custombatch" call.
// Exactly one of *DatafeedsCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *DatafeedsCustomBatchResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedsCustombatchCall) Do(opts ...googleapi.CallOption) (*DatafeedsCustomBatchResponse, error) {
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
	ret := &DatafeedsCustomBatchResponse{
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
	//   "httpMethod": "POST",
	//   "id": "content.datafeeds.custombatch",
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "datafeeds/batch",
	//   "request": {
	//     "$ref": "DatafeedsCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "DatafeedsCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeeds.delete":

type DatafeedsDeleteCall struct {
	s          *APIService
	merchantId uint64
	datafeedId uint64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a datafeed configuration from your Merchant Center
// account. This method can only be called for non-multi-client
// accounts.
func (r *DatafeedsService) Delete(merchantId uint64, datafeedId uint64) *DatafeedsDeleteCall {
	c := &DatafeedsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeedId = datafeedId
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *DatafeedsDeleteCall) DryRun(dryRun bool) *DatafeedsDeleteCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedsDeleteCall) Fields(s ...googleapi.Field) *DatafeedsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedsDeleteCall) Context(ctx context.Context) *DatafeedsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeeds.delete" call.
func (c *DatafeedsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Deletes a datafeed configuration from your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "DELETE",
	//   "id": "content.datafeeds.delete",
	//   "parameterOrder": [
	//     "merchantId",
	//     "datafeedId"
	//   ],
	//   "parameters": {
	//     "datafeedId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeeds/{datafeedId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeeds.get":

type DatafeedsGetCall struct {
	s            *APIService
	merchantId   uint64
	datafeedId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a datafeed configuration from your Merchant Center
// account. This method can only be called for non-multi-client
// accounts.
func (r *DatafeedsService) Get(merchantId uint64, datafeedId uint64) *DatafeedsGetCall {
	c := &DatafeedsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeedId = datafeedId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedsGetCall) Fields(s ...googleapi.Field) *DatafeedsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatafeedsGetCall) IfNoneMatch(entityTag string) *DatafeedsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedsGetCall) Context(ctx context.Context) *DatafeedsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeeds.get" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsGetCall) Do(opts ...googleapi.CallOption) (*Datafeed, error) {
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
	ret := &Datafeed{
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
	//   "description": "Retrieves a datafeed configuration from your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.datafeeds.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "datafeedId"
	//   ],
	//   "parameters": {
	//     "datafeedId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeeds/{datafeedId}",
	//   "response": {
	//     "$ref": "Datafeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeeds.insert":

type DatafeedsInsertCall struct {
	s          *APIService
	merchantId uint64
	datafeed   *Datafeed
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Registers a datafeed configuration with your Merchant Center
// account. This method can only be called for non-multi-client
// accounts.
func (r *DatafeedsService) Insert(merchantId uint64, datafeed *Datafeed) *DatafeedsInsertCall {
	c := &DatafeedsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeed = datafeed
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *DatafeedsInsertCall) DryRun(dryRun bool) *DatafeedsInsertCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedsInsertCall) Fields(s ...googleapi.Field) *DatafeedsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedsInsertCall) Context(ctx context.Context) *DatafeedsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeed)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeeds.insert" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsInsertCall) Do(opts ...googleapi.CallOption) (*Datafeed, error) {
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
	ret := &Datafeed{
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
	//   "description": "Registers a datafeed configuration with your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.datafeeds.insert",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeeds",
	//   "request": {
	//     "$ref": "Datafeed"
	//   },
	//   "response": {
	//     "$ref": "Datafeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeeds.list":

type DatafeedsListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the datafeeds in your Merchant Center account. This
// method can only be called for non-multi-client accounts.
func (r *DatafeedsService) List(merchantId uint64) *DatafeedsListCall {
	c := &DatafeedsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of products to return in the response, used for paging.
func (c *DatafeedsListCall) MaxResults(maxResults int64) *DatafeedsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *DatafeedsListCall) PageToken(pageToken string) *DatafeedsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedsListCall) Fields(s ...googleapi.Field) *DatafeedsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatafeedsListCall) IfNoneMatch(entityTag string) *DatafeedsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedsListCall) Context(ctx context.Context) *DatafeedsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeeds.list" call.
// Exactly one of *DatafeedsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DatafeedsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedsListCall) Do(opts ...googleapi.CallOption) (*DatafeedsListResponse, error) {
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
	ret := &DatafeedsListResponse{
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
	//   "description": "Lists the datafeeds in your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.datafeeds.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of products to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeeds",
	//   "response": {
	//     "$ref": "DatafeedsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *DatafeedsListCall) Pages(ctx context.Context, f func(*DatafeedsListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.datafeeds.patch":

type DatafeedsPatchCall struct {
	s          *APIService
	merchantId uint64
	datafeedId uint64
	datafeed   *Datafeed
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a datafeed configuration of your Merchant Center
// account. This method can only be called for non-multi-client
// accounts. This method supports patch semantics.
func (r *DatafeedsService) Patch(merchantId uint64, datafeedId uint64, datafeed *Datafeed) *DatafeedsPatchCall {
	c := &DatafeedsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeedId = datafeedId
	c.datafeed = datafeed
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *DatafeedsPatchCall) DryRun(dryRun bool) *DatafeedsPatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedsPatchCall) Fields(s ...googleapi.Field) *DatafeedsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedsPatchCall) Context(ctx context.Context) *DatafeedsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeed)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeeds.patch" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsPatchCall) Do(opts ...googleapi.CallOption) (*Datafeed, error) {
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
	ret := &Datafeed{
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
	//   "description": "Updates a datafeed configuration of your Merchant Center account. This method can only be called for non-multi-client accounts. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "content.datafeeds.patch",
	//   "parameterOrder": [
	//     "merchantId",
	//     "datafeedId"
	//   ],
	//   "parameters": {
	//     "datafeedId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeeds/{datafeedId}",
	//   "request": {
	//     "$ref": "Datafeed"
	//   },
	//   "response": {
	//     "$ref": "Datafeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeeds.update":

type DatafeedsUpdateCall struct {
	s          *APIService
	merchantId uint64
	datafeedId uint64
	datafeed   *Datafeed
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a datafeed configuration of your Merchant Center
// account. This method can only be called for non-multi-client
// accounts.
func (r *DatafeedsService) Update(merchantId uint64, datafeedId uint64, datafeed *Datafeed) *DatafeedsUpdateCall {
	c := &DatafeedsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeedId = datafeedId
	c.datafeed = datafeed
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *DatafeedsUpdateCall) DryRun(dryRun bool) *DatafeedsUpdateCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedsUpdateCall) Fields(s ...googleapi.Field) *DatafeedsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedsUpdateCall) Context(ctx context.Context) *DatafeedsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeed)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeeds.update" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsUpdateCall) Do(opts ...googleapi.CallOption) (*Datafeed, error) {
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
	ret := &Datafeed{
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
	//   "description": "Updates a datafeed configuration of your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "PUT",
	//   "id": "content.datafeeds.update",
	//   "parameterOrder": [
	//     "merchantId",
	//     "datafeedId"
	//   ],
	//   "parameters": {
	//     "datafeedId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeeds/{datafeedId}",
	//   "request": {
	//     "$ref": "Datafeed"
	//   },
	//   "response": {
	//     "$ref": "Datafeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeedstatuses.custombatch":

type DatafeedstatusesCustombatchCall struct {
	s                                  *APIService
	datafeedstatusescustombatchrequest *DatafeedstatusesCustomBatchRequest
	urlParams_                         gensupport.URLParams
	ctx_                               context.Context
	header_                            http.Header
}

// Custombatch:
func (r *DatafeedstatusesService) Custombatch(datafeedstatusescustombatchrequest *DatafeedstatusesCustomBatchRequest) *DatafeedstatusesCustombatchCall {
	c := &DatafeedstatusesCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datafeedstatusescustombatchrequest = datafeedstatusescustombatchrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedstatusesCustombatchCall) Fields(s ...googleapi.Field) *DatafeedstatusesCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedstatusesCustombatchCall) Context(ctx context.Context) *DatafeedstatusesCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedstatusesCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedstatusesCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeedstatusescustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "datafeedstatuses/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeedstatuses.custombatch" call.
// Exactly one of *DatafeedstatusesCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *DatafeedstatusesCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *DatafeedstatusesCustombatchCall) Do(opts ...googleapi.CallOption) (*DatafeedstatusesCustomBatchResponse, error) {
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
	ret := &DatafeedstatusesCustomBatchResponse{
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
	//   "httpMethod": "POST",
	//   "id": "content.datafeedstatuses.custombatch",
	//   "path": "datafeedstatuses/batch",
	//   "request": {
	//     "$ref": "DatafeedstatusesCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "DatafeedstatusesCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeedstatuses.get":

type DatafeedstatusesGetCall struct {
	s            *APIService
	merchantId   uint64
	datafeedId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the status of a datafeed from your Merchant Center
// account. This method can only be called for non-multi-client
// accounts.
func (r *DatafeedstatusesService) Get(merchantId uint64, datafeedId uint64) *DatafeedstatusesGetCall {
	c := &DatafeedstatusesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeedId = datafeedId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedstatusesGetCall) Fields(s ...googleapi.Field) *DatafeedstatusesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatafeedstatusesGetCall) IfNoneMatch(entityTag string) *DatafeedstatusesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedstatusesGetCall) Context(ctx context.Context) *DatafeedstatusesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedstatusesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedstatusesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeedstatuses/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeedstatuses.get" call.
// Exactly one of *DatafeedStatus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DatafeedStatus.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedstatusesGetCall) Do(opts ...googleapi.CallOption) (*DatafeedStatus, error) {
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
	ret := &DatafeedStatus{
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
	//   "description": "Retrieves the status of a datafeed from your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.datafeedstatuses.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "datafeedId"
	//   ],
	//   "parameters": {
	//     "datafeedId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantId": {
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeedstatuses/{datafeedId}",
	//   "response": {
	//     "$ref": "DatafeedStatus"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.datafeedstatuses.list":

type DatafeedstatusesListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the statuses of the datafeeds in your Merchant Center
// account. This method can only be called for non-multi-client
// accounts.
func (r *DatafeedstatusesService) List(merchantId uint64) *DatafeedstatusesListCall {
	c := &DatafeedstatusesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of products to return in the response, used for paging.
func (c *DatafeedstatusesListCall) MaxResults(maxResults int64) *DatafeedstatusesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *DatafeedstatusesListCall) PageToken(pageToken string) *DatafeedstatusesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatafeedstatusesListCall) Fields(s ...googleapi.Field) *DatafeedstatusesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatafeedstatusesListCall) IfNoneMatch(entityTag string) *DatafeedstatusesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatafeedstatusesListCall) Context(ctx context.Context) *DatafeedstatusesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatafeedstatusesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatafeedstatusesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeedstatuses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.datafeedstatuses.list" call.
// Exactly one of *DatafeedstatusesListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *DatafeedstatusesListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedstatusesListCall) Do(opts ...googleapi.CallOption) (*DatafeedstatusesListResponse, error) {
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
	ret := &DatafeedstatusesListResponse{
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
	//   "description": "Lists the statuses of the datafeeds in your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.datafeedstatuses.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of products to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/datafeedstatuses",
	//   "response": {
	//     "$ref": "DatafeedstatusesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *DatafeedstatusesListCall) Pages(ctx context.Context, f func(*DatafeedstatusesListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.inventory.custombatch":

type InventoryCustombatchCall struct {
	s                           *APIService
	inventorycustombatchrequest *InventoryCustomBatchRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Custombatch: Updates price and availability for multiple products or
// stores in a single request. This operation does not update the
// expiration date of the products. This method can only be called for
// non-multi-client accounts.
func (r *InventoryService) Custombatch(inventorycustombatchrequest *InventoryCustomBatchRequest) *InventoryCustombatchCall {
	c := &InventoryCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.inventorycustombatchrequest = inventorycustombatchrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *InventoryCustombatchCall) DryRun(dryRun bool) *InventoryCustombatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InventoryCustombatchCall) Fields(s ...googleapi.Field) *InventoryCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InventoryCustombatchCall) Context(ctx context.Context) *InventoryCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InventoryCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InventoryCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inventorycustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "inventory/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.inventory.custombatch" call.
// Exactly one of *InventoryCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *InventoryCustomBatchResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InventoryCustombatchCall) Do(opts ...googleapi.CallOption) (*InventoryCustomBatchResponse, error) {
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
	ret := &InventoryCustomBatchResponse{
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
	//   "description": "Updates price and availability for multiple products or stores in a single request. This operation does not update the expiration date of the products. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.inventory.custombatch",
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "inventory/batch",
	//   "request": {
	//     "$ref": "InventoryCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "InventoryCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.inventory.set":

type InventorySetCall struct {
	s                   *APIService
	merchantId          uint64
	storeCode           string
	productId           string
	inventorysetrequest *InventorySetRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// Set: Updates price and availability of a product in your Merchant
// Center account. This operation does not update the expiration date of
// the product. This method can only be called for non-multi-client
// accounts.
func (r *InventoryService) Set(merchantId uint64, storeCode string, productId string, inventorysetrequest *InventorySetRequest) *InventorySetCall {
	c := &InventorySetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.storeCode = storeCode
	c.productId = productId
	c.inventorysetrequest = inventorysetrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *InventorySetCall) DryRun(dryRun bool) *InventorySetCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InventorySetCall) Fields(s ...googleapi.Field) *InventorySetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InventorySetCall) Context(ctx context.Context) *InventorySetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InventorySetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InventorySetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inventorysetrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/inventory/{storeCode}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"storeCode":  c.storeCode,
		"productId":  c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.inventory.set" call.
// Exactly one of *InventorySetResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *InventorySetResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InventorySetCall) Do(opts ...googleapi.CallOption) (*InventorySetResponse, error) {
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
	ret := &InventorySetResponse{
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
	//   "description": "Updates price and availability of a product in your Merchant Center account. This operation does not update the expiration date of the product. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.inventory.set",
	//   "parameterOrder": [
	//     "merchantId",
	//     "storeCode",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product for which to update price and availability.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "storeCode": {
	//       "description": "The code of the store for which to update price and availability. Use online to update price and availability of an online product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/inventory/{storeCode}/products/{productId}",
	//   "request": {
	//     "$ref": "InventorySetRequest"
	//   },
	//   "response": {
	//     "$ref": "InventorySetResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.acknowledge":

type OrdersAcknowledgeCall struct {
	s                        *APIService
	merchantId               uint64
	orderId                  string
	ordersacknowledgerequest *OrdersAcknowledgeRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// Acknowledge: Marks an order as acknowledged. This method can only be
// called for non-multi-client accounts.
func (r *OrdersService) Acknowledge(merchantId uint64, orderId string, ordersacknowledgerequest *OrdersAcknowledgeRequest) *OrdersAcknowledgeCall {
	c := &OrdersAcknowledgeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersacknowledgerequest = ordersacknowledgerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersAcknowledgeCall) Fields(s ...googleapi.Field) *OrdersAcknowledgeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersAcknowledgeCall) Context(ctx context.Context) *OrdersAcknowledgeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersAcknowledgeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersAcknowledgeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersacknowledgerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/acknowledge")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.acknowledge" call.
// Exactly one of *OrdersAcknowledgeResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *OrdersAcknowledgeResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersAcknowledgeCall) Do(opts ...googleapi.CallOption) (*OrdersAcknowledgeResponse, error) {
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
	ret := &OrdersAcknowledgeResponse{
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
	//   "description": "Marks an order as acknowledged. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.acknowledge",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/acknowledge",
	//   "request": {
	//     "$ref": "OrdersAcknowledgeRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersAcknowledgeResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.advancetestorder":

type OrdersAdvancetestorderCall struct {
	s          *APIService
	merchantId uint64
	orderId    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Advancetestorder: Sandbox only. Moves a test order from state
// "inProgress" to state "pendingShipment". This method can only be
// called for non-multi-client accounts.
func (r *OrdersService) Advancetestorder(merchantId uint64, orderId string) *OrdersAdvancetestorderCall {
	c := &OrdersAdvancetestorderCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersAdvancetestorderCall) Fields(s ...googleapi.Field) *OrdersAdvancetestorderCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersAdvancetestorderCall) Context(ctx context.Context) *OrdersAdvancetestorderCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersAdvancetestorderCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersAdvancetestorderCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/testorders/{orderId}/advance")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.advancetestorder" call.
// Exactly one of *OrdersAdvanceTestOrderResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersAdvanceTestOrderResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersAdvancetestorderCall) Do(opts ...googleapi.CallOption) (*OrdersAdvanceTestOrderResponse, error) {
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
	ret := &OrdersAdvanceTestOrderResponse{
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
	//   "description": "Sandbox only. Moves a test order from state \"inProgress\" to state \"pendingShipment\". This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.advancetestorder",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the test order to modify.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/testorders/{orderId}/advance",
	//   "response": {
	//     "$ref": "OrdersAdvanceTestOrderResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.cancel":

type OrdersCancelCall struct {
	s                   *APIService
	merchantId          uint64
	orderId             string
	orderscancelrequest *OrdersCancelRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// Cancel: Cancels all line items in an order, making a full refund.
// This method can only be called for non-multi-client accounts.
func (r *OrdersService) Cancel(merchantId uint64, orderId string, orderscancelrequest *OrdersCancelRequest) *OrdersCancelCall {
	c := &OrdersCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.orderscancelrequest = orderscancelrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersCancelCall) Fields(s ...googleapi.Field) *OrdersCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersCancelCall) Context(ctx context.Context) *OrdersCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscancelrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.cancel" call.
// Exactly one of *OrdersCancelResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OrdersCancelResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCancelCall) Do(opts ...googleapi.CallOption) (*OrdersCancelResponse, error) {
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
	ret := &OrdersCancelResponse{
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
	//   "description": "Cancels all line items in an order, making a full refund. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.cancel",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order to cancel.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/cancel",
	//   "request": {
	//     "$ref": "OrdersCancelRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersCancelResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.cancellineitem":

type OrdersCancellineitemCall struct {
	s                           *APIService
	merchantId                  uint64
	orderId                     string
	orderscancellineitemrequest *OrdersCancelLineItemRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Cancellineitem: Cancels a line item, making a full refund. This
// method can only be called for non-multi-client accounts.
func (r *OrdersService) Cancellineitem(merchantId uint64, orderId string, orderscancellineitemrequest *OrdersCancelLineItemRequest) *OrdersCancellineitemCall {
	c := &OrdersCancellineitemCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.orderscancellineitemrequest = orderscancellineitemrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersCancellineitemCall) Fields(s ...googleapi.Field) *OrdersCancellineitemCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersCancellineitemCall) Context(ctx context.Context) *OrdersCancellineitemCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersCancellineitemCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersCancellineitemCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscancellineitemrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/cancelLineItem")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.cancellineitem" call.
// Exactly one of *OrdersCancelLineItemResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersCancelLineItemResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCancellineitemCall) Do(opts ...googleapi.CallOption) (*OrdersCancelLineItemResponse, error) {
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
	ret := &OrdersCancelLineItemResponse{
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
	//   "description": "Cancels a line item, making a full refund. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.cancellineitem",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/cancelLineItem",
	//   "request": {
	//     "$ref": "OrdersCancelLineItemRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersCancelLineItemResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.createtestorder":

type OrdersCreatetestorderCall struct {
	s                            *APIService
	merchantId                   uint64
	orderscreatetestorderrequest *OrdersCreateTestOrderRequest
	urlParams_                   gensupport.URLParams
	ctx_                         context.Context
	header_                      http.Header
}

// Createtestorder: Sandbox only. Creates a test order. This method can
// only be called for non-multi-client accounts.
func (r *OrdersService) Createtestorder(merchantId uint64, orderscreatetestorderrequest *OrdersCreateTestOrderRequest) *OrdersCreatetestorderCall {
	c := &OrdersCreatetestorderCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderscreatetestorderrequest = orderscreatetestorderrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersCreatetestorderCall) Fields(s ...googleapi.Field) *OrdersCreatetestorderCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersCreatetestorderCall) Context(ctx context.Context) *OrdersCreatetestorderCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersCreatetestorderCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersCreatetestorderCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscreatetestorderrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/testorders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.createtestorder" call.
// Exactly one of *OrdersCreateTestOrderResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersCreateTestOrderResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCreatetestorderCall) Do(opts ...googleapi.CallOption) (*OrdersCreateTestOrderResponse, error) {
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
	ret := &OrdersCreateTestOrderResponse{
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
	//   "description": "Sandbox only. Creates a test order. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.createtestorder",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/testorders",
	//   "request": {
	//     "$ref": "OrdersCreateTestOrderRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersCreateTestOrderResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.custombatch":

type OrdersCustombatchCall struct {
	s                        *APIService
	orderscustombatchrequest *OrdersCustomBatchRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// Custombatch: Retrieves or modifies multiple orders in a single
// request. This method can only be called for non-multi-client
// accounts.
func (r *OrdersService) Custombatch(orderscustombatchrequest *OrdersCustomBatchRequest) *OrdersCustombatchCall {
	c := &OrdersCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderscustombatchrequest = orderscustombatchrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersCustombatchCall) Fields(s ...googleapi.Field) *OrdersCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersCustombatchCall) Context(ctx context.Context) *OrdersCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "orders/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.custombatch" call.
// Exactly one of *OrdersCustomBatchResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *OrdersCustomBatchResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCustombatchCall) Do(opts ...googleapi.CallOption) (*OrdersCustomBatchResponse, error) {
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
	ret := &OrdersCustomBatchResponse{
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
	//   "description": "Retrieves or modifies multiple orders in a single request. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.custombatch",
	//   "path": "orders/batch",
	//   "request": {
	//     "$ref": "OrdersCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.get":

type OrdersGetCall struct {
	s            *APIService
	merchantId   uint64
	orderId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves an order from your Merchant Center account. This
// method can only be called for non-multi-client accounts.
func (r *OrdersService) Get(merchantId uint64, orderId string) *OrdersGetCall {
	c := &OrdersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersGetCall) Fields(s ...googleapi.Field) *OrdersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OrdersGetCall) IfNoneMatch(entityTag string) *OrdersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersGetCall) Context(ctx context.Context) *OrdersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.get" call.
// Exactly one of *Order or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Order.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OrdersGetCall) Do(opts ...googleapi.CallOption) (*Order, error) {
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
	ret := &Order{
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
	//   "description": "Retrieves an order from your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.orders.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}",
	//   "response": {
	//     "$ref": "Order"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.getbymerchantorderid":

type OrdersGetbymerchantorderidCall struct {
	s               *APIService
	merchantId      uint64
	merchantOrderId string
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// Getbymerchantorderid: Retrieves an order using merchant order id.
// This method can only be called for non-multi-client accounts.
func (r *OrdersService) Getbymerchantorderid(merchantId uint64, merchantOrderId string) *OrdersGetbymerchantorderidCall {
	c := &OrdersGetbymerchantorderidCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.merchantOrderId = merchantOrderId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersGetbymerchantorderidCall) Fields(s ...googleapi.Field) *OrdersGetbymerchantorderidCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OrdersGetbymerchantorderidCall) IfNoneMatch(entityTag string) *OrdersGetbymerchantorderidCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersGetbymerchantorderidCall) Context(ctx context.Context) *OrdersGetbymerchantorderidCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersGetbymerchantorderidCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersGetbymerchantorderidCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/ordersbymerchantid/{merchantOrderId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId":      strconv.FormatUint(c.merchantId, 10),
		"merchantOrderId": c.merchantOrderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.getbymerchantorderid" call.
// Exactly one of *OrdersGetByMerchantOrderIdResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersGetByMerchantOrderIdResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrdersGetbymerchantorderidCall) Do(opts ...googleapi.CallOption) (*OrdersGetByMerchantOrderIdResponse, error) {
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
	ret := &OrdersGetByMerchantOrderIdResponse{
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
	//   "description": "Retrieves an order using merchant order id. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.orders.getbymerchantorderid",
	//   "parameterOrder": [
	//     "merchantId",
	//     "merchantOrderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantOrderId": {
	//       "description": "The merchant order id to be looked for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/ordersbymerchantid/{merchantOrderId}",
	//   "response": {
	//     "$ref": "OrdersGetByMerchantOrderIdResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.gettestordertemplate":

type OrdersGettestordertemplateCall struct {
	s            *APIService
	merchantId   uint64
	templateName string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Gettestordertemplate: Sandbox only. Retrieves an order template that
// can be used to quickly create a new order in sandbox. This method can
// only be called for non-multi-client accounts.
func (r *OrdersService) Gettestordertemplate(merchantId uint64, templateName string) *OrdersGettestordertemplateCall {
	c := &OrdersGettestordertemplateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.templateName = templateName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersGettestordertemplateCall) Fields(s ...googleapi.Field) *OrdersGettestordertemplateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OrdersGettestordertemplateCall) IfNoneMatch(entityTag string) *OrdersGettestordertemplateCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersGettestordertemplateCall) Context(ctx context.Context) *OrdersGettestordertemplateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersGettestordertemplateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersGettestordertemplateCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/testordertemplates/{templateName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId":   strconv.FormatUint(c.merchantId, 10),
		"templateName": c.templateName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.gettestordertemplate" call.
// Exactly one of *OrdersGetTestOrderTemplateResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersGetTestOrderTemplateResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrdersGettestordertemplateCall) Do(opts ...googleapi.CallOption) (*OrdersGetTestOrderTemplateResponse, error) {
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
	ret := &OrdersGetTestOrderTemplateResponse{
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
	//   "description": "Sandbox only. Retrieves an order template that can be used to quickly create a new order in sandbox. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.orders.gettestordertemplate",
	//   "parameterOrder": [
	//     "merchantId",
	//     "templateName"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "templateName": {
	//       "description": "The name of the template to retrieve.",
	//       "enum": [
	//         "template1",
	//         "template1a",
	//         "template1b",
	//         "template2"
	//       ],
	//       "enumDescriptions": [
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
	//   "path": "{merchantId}/testordertemplates/{templateName}",
	//   "response": {
	//     "$ref": "OrdersGetTestOrderTemplateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.list":

type OrdersListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the orders in your Merchant Center account. This method
// can only be called for non-multi-client accounts.
func (r *OrdersService) List(merchantId uint64) *OrdersListCall {
	c := &OrdersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// Acknowledged sets the optional parameter "acknowledged": Obtains
// orders that match the acknowledgement status. When set to true,
// obtains orders that have been acknowledged. When false, obtains
// orders that have not been acknowledged.
// We recommend using this filter set to false, in conjunction with the
// acknowledge call, such that only un-acknowledged orders are returned.
func (c *OrdersListCall) Acknowledged(acknowledged bool) *OrdersListCall {
	c.urlParams_.Set("acknowledged", fmt.Sprint(acknowledged))
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of orders to return in the response, used for paging. The
// default value is 25 orders per page, and the maximum allowed value is
// 250 orders per page.
// Known issue: All List calls will return all Orders without limit
// regardless of the value of this field.
func (c *OrdersListCall) MaxResults(maxResults int64) *OrdersListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// OrderBy sets the optional parameter "orderBy": The ordering of the
// returned list. The only supported value are placedDate desc and
// placedDate asc for now, which returns orders sorted by placement
// date. "placedDate desc" stands for listing orders by placement date,
// from oldest to most recent. "placedDate asc" stands for listing
// orders by placement date, from most recent to oldest. In future
// releases we'll support other sorting criteria.
//
// Possible values:
//   "placedDate asc"
//   "placedDate desc"
func (c *OrdersListCall) OrderBy(orderBy string) *OrdersListCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *OrdersListCall) PageToken(pageToken string) *OrdersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// PlacedDateEnd sets the optional parameter "placedDateEnd": Obtains
// orders placed before this date (exclusively), in ISO 8601 format.
func (c *OrdersListCall) PlacedDateEnd(placedDateEnd string) *OrdersListCall {
	c.urlParams_.Set("placedDateEnd", placedDateEnd)
	return c
}

// PlacedDateStart sets the optional parameter "placedDateStart":
// Obtains orders placed after this date (inclusively), in ISO 8601
// format.
func (c *OrdersListCall) PlacedDateStart(placedDateStart string) *OrdersListCall {
	c.urlParams_.Set("placedDateStart", placedDateStart)
	return c
}

// Statuses sets the optional parameter "statuses": Obtains orders that
// match any of the specified statuses. Multiple values can be specified
// with comma separation. Additionally, please note that active is a
// shortcut for pendingShipment and partiallyShipped, and completed is a
// shortcut for shipped , partiallyDelivered, delivered,
// partiallyReturned, returned, and canceled.
//
// Possible values:
//   "active"
//   "canceled"
//   "completed"
//   "delivered"
//   "inProgress"
//   "partiallyDelivered"
//   "partiallyReturned"
//   "partiallyShipped"
//   "pendingShipment"
//   "returned"
//   "shipped"
func (c *OrdersListCall) Statuses(statuses ...string) *OrdersListCall {
	c.urlParams_.SetMulti("statuses", append([]string{}, statuses...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersListCall) Fields(s ...googleapi.Field) *OrdersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OrdersListCall) IfNoneMatch(entityTag string) *OrdersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersListCall) Context(ctx context.Context) *OrdersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.list" call.
// Exactly one of *OrdersListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OrdersListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersListCall) Do(opts ...googleapi.CallOption) (*OrdersListResponse, error) {
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
	ret := &OrdersListResponse{
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
	//   "description": "Lists the orders in your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.orders.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "acknowledged": {
	//       "description": "Obtains orders that match the acknowledgement status. When set to true, obtains orders that have been acknowledged. When false, obtains orders that have not been acknowledged.\nWe recommend using this filter set to false, in conjunction with the acknowledge call, such that only un-acknowledged orders are returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of orders to return in the response, used for paging. The default value is 25 orders per page, and the maximum allowed value is 250 orders per page.\nKnown issue: All List calls will return all Orders without limit regardless of the value of this field.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderBy": {
	//       "description": "The ordering of the returned list. The only supported value are placedDate desc and placedDate asc for now, which returns orders sorted by placement date. \"placedDate desc\" stands for listing orders by placement date, from oldest to most recent. \"placedDate asc\" stands for listing orders by placement date, from most recent to oldest. In future releases we'll support other sorting criteria.",
	//       "enum": [
	//         "placedDate asc",
	//         "placedDate desc"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "placedDateEnd": {
	//       "description": "Obtains orders placed before this date (exclusively), in ISO 8601 format.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "placedDateStart": {
	//       "description": "Obtains orders placed after this date (inclusively), in ISO 8601 format.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "statuses": {
	//       "description": "Obtains orders that match any of the specified statuses. Multiple values can be specified with comma separation. Additionally, please note that active is a shortcut for pendingShipment and partiallyShipped, and completed is a shortcut for shipped , partiallyDelivered, delivered, partiallyReturned, returned, and canceled.",
	//       "enum": [
	//         "active",
	//         "canceled",
	//         "completed",
	//         "delivered",
	//         "inProgress",
	//         "partiallyDelivered",
	//         "partiallyReturned",
	//         "partiallyShipped",
	//         "pendingShipment",
	//         "returned",
	//         "shipped"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders",
	//   "response": {
	//     "$ref": "OrdersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *OrdersListCall) Pages(ctx context.Context, f func(*OrdersListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.orders.refund":

type OrdersRefundCall struct {
	s                   *APIService
	merchantId          uint64
	orderId             string
	ordersrefundrequest *OrdersRefundRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// Refund: Refund a portion of the order, up to the full amount paid.
// This method can only be called for non-multi-client accounts.
func (r *OrdersService) Refund(merchantId uint64, orderId string, ordersrefundrequest *OrdersRefundRequest) *OrdersRefundCall {
	c := &OrdersRefundCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersrefundrequest = ordersrefundrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersRefundCall) Fields(s ...googleapi.Field) *OrdersRefundCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersRefundCall) Context(ctx context.Context) *OrdersRefundCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersRefundCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersRefundCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersrefundrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/refund")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.refund" call.
// Exactly one of *OrdersRefundResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OrdersRefundResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersRefundCall) Do(opts ...googleapi.CallOption) (*OrdersRefundResponse, error) {
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
	ret := &OrdersRefundResponse{
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
	//   "description": "Refund a portion of the order, up to the full amount paid. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.refund",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order to refund.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/refund",
	//   "request": {
	//     "$ref": "OrdersRefundRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersRefundResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.returnlineitem":

type OrdersReturnlineitemCall struct {
	s                           *APIService
	merchantId                  uint64
	orderId                     string
	ordersreturnlineitemrequest *OrdersReturnLineItemRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Returnlineitem: Returns a line item. This method can only be called
// for non-multi-client accounts.
func (r *OrdersService) Returnlineitem(merchantId uint64, orderId string, ordersreturnlineitemrequest *OrdersReturnLineItemRequest) *OrdersReturnlineitemCall {
	c := &OrdersReturnlineitemCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersreturnlineitemrequest = ordersreturnlineitemrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersReturnlineitemCall) Fields(s ...googleapi.Field) *OrdersReturnlineitemCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersReturnlineitemCall) Context(ctx context.Context) *OrdersReturnlineitemCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersReturnlineitemCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersReturnlineitemCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersreturnlineitemrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/returnLineItem")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.returnlineitem" call.
// Exactly one of *OrdersReturnLineItemResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersReturnLineItemResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersReturnlineitemCall) Do(opts ...googleapi.CallOption) (*OrdersReturnLineItemResponse, error) {
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
	ret := &OrdersReturnLineItemResponse{
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
	//   "description": "Returns a line item. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.returnlineitem",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/returnLineItem",
	//   "request": {
	//     "$ref": "OrdersReturnLineItemRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersReturnLineItemResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.shiplineitems":

type OrdersShiplineitemsCall struct {
	s                          *APIService
	merchantId                 uint64
	orderId                    string
	ordersshiplineitemsrequest *OrdersShipLineItemsRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// Shiplineitems: Marks line item(s) as shipped. This method can only be
// called for non-multi-client accounts.
func (r *OrdersService) Shiplineitems(merchantId uint64, orderId string, ordersshiplineitemsrequest *OrdersShipLineItemsRequest) *OrdersShiplineitemsCall {
	c := &OrdersShiplineitemsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersshiplineitemsrequest = ordersshiplineitemsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersShiplineitemsCall) Fields(s ...googleapi.Field) *OrdersShiplineitemsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersShiplineitemsCall) Context(ctx context.Context) *OrdersShiplineitemsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersShiplineitemsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersShiplineitemsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersshiplineitemsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/shipLineItems")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.shiplineitems" call.
// Exactly one of *OrdersShipLineItemsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *OrdersShipLineItemsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersShiplineitemsCall) Do(opts ...googleapi.CallOption) (*OrdersShipLineItemsResponse, error) {
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
	ret := &OrdersShipLineItemsResponse{
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
	//   "description": "Marks line item(s) as shipped. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.shiplineitems",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/shipLineItems",
	//   "request": {
	//     "$ref": "OrdersShipLineItemsRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersShipLineItemsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.updatemerchantorderid":

type OrdersUpdatemerchantorderidCall struct {
	s                                  *APIService
	merchantId                         uint64
	orderId                            string
	ordersupdatemerchantorderidrequest *OrdersUpdateMerchantOrderIdRequest
	urlParams_                         gensupport.URLParams
	ctx_                               context.Context
	header_                            http.Header
}

// Updatemerchantorderid: Updates the merchant order ID for a given
// order. This method can only be called for non-multi-client accounts.
func (r *OrdersService) Updatemerchantorderid(merchantId uint64, orderId string, ordersupdatemerchantorderidrequest *OrdersUpdateMerchantOrderIdRequest) *OrdersUpdatemerchantorderidCall {
	c := &OrdersUpdatemerchantorderidCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersupdatemerchantorderidrequest = ordersupdatemerchantorderidrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersUpdatemerchantorderidCall) Fields(s ...googleapi.Field) *OrdersUpdatemerchantorderidCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersUpdatemerchantorderidCall) Context(ctx context.Context) *OrdersUpdatemerchantorderidCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersUpdatemerchantorderidCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersUpdatemerchantorderidCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersupdatemerchantorderidrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/updateMerchantOrderId")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.updatemerchantorderid" call.
// Exactly one of *OrdersUpdateMerchantOrderIdResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersUpdateMerchantOrderIdResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrdersUpdatemerchantorderidCall) Do(opts ...googleapi.CallOption) (*OrdersUpdateMerchantOrderIdResponse, error) {
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
	ret := &OrdersUpdateMerchantOrderIdResponse{
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
	//   "description": "Updates the merchant order ID for a given order. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.updatemerchantorderid",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/updateMerchantOrderId",
	//   "request": {
	//     "$ref": "OrdersUpdateMerchantOrderIdRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersUpdateMerchantOrderIdResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.orders.updateshipment":

type OrdersUpdateshipmentCall struct {
	s                           *APIService
	merchantId                  uint64
	orderId                     string
	ordersupdateshipmentrequest *OrdersUpdateShipmentRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Updateshipment: Updates a shipment's status, carrier, and/or tracking
// ID. This method can only be called for non-multi-client accounts.
func (r *OrdersService) Updateshipment(merchantId uint64, orderId string, ordersupdateshipmentrequest *OrdersUpdateShipmentRequest) *OrdersUpdateshipmentCall {
	c := &OrdersUpdateshipmentCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersupdateshipmentrequest = ordersupdateshipmentrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrdersUpdateshipmentCall) Fields(s ...googleapi.Field) *OrdersUpdateshipmentCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrdersUpdateshipmentCall) Context(ctx context.Context) *OrdersUpdateshipmentCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrdersUpdateshipmentCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrdersUpdateshipmentCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersupdateshipmentrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/updateShipment")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.orders.updateshipment" call.
// Exactly one of *OrdersUpdateShipmentResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersUpdateShipmentResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersUpdateshipmentCall) Do(opts ...googleapi.CallOption) (*OrdersUpdateShipmentResponse, error) {
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
	ret := &OrdersUpdateShipmentResponse{
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
	//   "description": "Updates a shipment's status, carrier, and/or tracking ID. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.orders.updateshipment",
	//   "parameterOrder": [
	//     "merchantId",
	//     "orderId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "The ID of the order.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/orders/{orderId}/updateShipment",
	//   "request": {
	//     "$ref": "OrdersUpdateShipmentRequest"
	//   },
	//   "response": {
	//     "$ref": "OrdersUpdateShipmentResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.products.custombatch":

type ProductsCustombatchCall struct {
	s                          *APIService
	productscustombatchrequest *ProductsCustomBatchRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// Custombatch: Retrieves, inserts, and deletes multiple products in a
// single request. This method can only be called for non-multi-client
// accounts.
func (r *ProductsService) Custombatch(productscustombatchrequest *ProductsCustomBatchRequest) *ProductsCustombatchCall {
	c := &ProductsCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productscustombatchrequest = productscustombatchrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *ProductsCustombatchCall) DryRun(dryRun bool) *ProductsCustombatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsCustombatchCall) Fields(s ...googleapi.Field) *ProductsCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsCustombatchCall) Context(ctx context.Context) *ProductsCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productscustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "products/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.products.custombatch" call.
// Exactly one of *ProductsCustomBatchResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ProductsCustomBatchResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsCustombatchCall) Do(opts ...googleapi.CallOption) (*ProductsCustomBatchResponse, error) {
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
	ret := &ProductsCustomBatchResponse{
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
	//   "description": "Retrieves, inserts, and deletes multiple products in a single request. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.products.custombatch",
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "products/batch",
	//   "request": {
	//     "$ref": "ProductsCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "ProductsCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.products.delete":

type ProductsDeleteCall struct {
	s          *APIService
	merchantId uint64
	productId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a product from your Merchant Center account. This
// method can only be called for non-multi-client accounts.
func (r *ProductsService) Delete(merchantId uint64, productId string) *ProductsDeleteCall {
	c := &ProductsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.productId = productId
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *ProductsDeleteCall) DryRun(dryRun bool) *ProductsDeleteCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsDeleteCall) Fields(s ...googleapi.Field) *ProductsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsDeleteCall) Context(ctx context.Context) *ProductsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"productId":  c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.products.delete" call.
func (c *ProductsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	//   "description": "Deletes a product from your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "DELETE",
	//   "id": "content.products.delete",
	//   "parameterOrder": [
	//     "merchantId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/products/{productId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.products.get":

type ProductsGetCall struct {
	s            *APIService
	merchantId   uint64
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a product from your Merchant Center account. This
// method can only be called for non-multi-client accounts.
func (r *ProductsService) Get(merchantId uint64, productId string) *ProductsGetCall {
	c := &ProductsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.productId = productId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsGetCall) Fields(s ...googleapi.Field) *ProductsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductsGetCall) IfNoneMatch(entityTag string) *ProductsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsGetCall) Context(ctx context.Context) *ProductsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"productId":  c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.products.get" call.
// Exactly one of *Product or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Product.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProductsGetCall) Do(opts ...googleapi.CallOption) (*Product, error) {
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
	ret := &Product{
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
	//   "description": "Retrieves a product from your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.products.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/products/{productId}",
	//   "response": {
	//     "$ref": "Product"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.products.insert":

type ProductsInsertCall struct {
	s          *APIService
	merchantId uint64
	product    *Product
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Uploads a product to your Merchant Center account. If an item
// with the same channel, contentLanguage, offerId, and targetCountry
// already exists, this method updates that entry. This method can only
// be called for non-multi-client accounts.
func (r *ProductsService) Insert(merchantId uint64, product *Product) *ProductsInsertCall {
	c := &ProductsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.product = product
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *ProductsInsertCall) DryRun(dryRun bool) *ProductsInsertCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsInsertCall) Fields(s ...googleapi.Field) *ProductsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsInsertCall) Context(ctx context.Context) *ProductsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.product)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.products.insert" call.
// Exactly one of *Product or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Product.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProductsInsertCall) Do(opts ...googleapi.CallOption) (*Product, error) {
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
	ret := &Product{
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
	//   "description": "Uploads a product to your Merchant Center account. If an item with the same channel, contentLanguage, offerId, and targetCountry already exists, this method updates that entry. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.products.insert",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/products",
	//   "request": {
	//     "$ref": "Product"
	//   },
	//   "response": {
	//     "$ref": "Product"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.products.list":

type ProductsListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the products in your Merchant Center account. This method
// can only be called for non-multi-client accounts.
func (r *ProductsService) List(merchantId uint64) *ProductsListCall {
	c := &ProductsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// IncludeInvalidInsertedItems sets the optional parameter
// "includeInvalidInsertedItems": Flag to include the invalid inserted
// items in the result of the list request. By default the invalid items
// are not shown (the default value is false).
func (c *ProductsListCall) IncludeInvalidInsertedItems(includeInvalidInsertedItems bool) *ProductsListCall {
	c.urlParams_.Set("includeInvalidInsertedItems", fmt.Sprint(includeInvalidInsertedItems))
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of products to return in the response, used for paging.
func (c *ProductsListCall) MaxResults(maxResults int64) *ProductsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *ProductsListCall) PageToken(pageToken string) *ProductsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsListCall) Fields(s ...googleapi.Field) *ProductsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductsListCall) IfNoneMatch(entityTag string) *ProductsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsListCall) Context(ctx context.Context) *ProductsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.products.list" call.
// Exactly one of *ProductsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ProductsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsListCall) Do(opts ...googleapi.CallOption) (*ProductsListResponse, error) {
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
	ret := &ProductsListResponse{
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
	//   "description": "Lists the products in your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.products.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "includeInvalidInsertedItems": {
	//       "description": "Flag to include the invalid inserted items in the result of the list request. By default the invalid items are not shown (the default value is false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of products to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/products",
	//   "response": {
	//     "$ref": "ProductsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProductsListCall) Pages(ctx context.Context, f func(*ProductsListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.productstatuses.custombatch":

type ProductstatusesCustombatchCall struct {
	s                                 *APIService
	productstatusescustombatchrequest *ProductstatusesCustomBatchRequest
	urlParams_                        gensupport.URLParams
	ctx_                              context.Context
	header_                           http.Header
}

// Custombatch: Gets the statuses of multiple products in a single
// request. This method can only be called for non-multi-client
// accounts.
func (r *ProductstatusesService) Custombatch(productstatusescustombatchrequest *ProductstatusesCustomBatchRequest) *ProductstatusesCustombatchCall {
	c := &ProductstatusesCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productstatusescustombatchrequest = productstatusescustombatchrequest
	return c
}

// IncludeAttributes sets the optional parameter "includeAttributes":
// Flag to include full product data in the results of this request. The
// default value is false.
func (c *ProductstatusesCustombatchCall) IncludeAttributes(includeAttributes bool) *ProductstatusesCustombatchCall {
	c.urlParams_.Set("includeAttributes", fmt.Sprint(includeAttributes))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductstatusesCustombatchCall) Fields(s ...googleapi.Field) *ProductstatusesCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductstatusesCustombatchCall) Context(ctx context.Context) *ProductstatusesCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductstatusesCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductstatusesCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productstatusescustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "productstatuses/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.productstatuses.custombatch" call.
// Exactly one of *ProductstatusesCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ProductstatusesCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProductstatusesCustombatchCall) Do(opts ...googleapi.CallOption) (*ProductstatusesCustomBatchResponse, error) {
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
	ret := &ProductstatusesCustomBatchResponse{
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
	//   "description": "Gets the statuses of multiple products in a single request. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "POST",
	//   "id": "content.productstatuses.custombatch",
	//   "parameters": {
	//     "includeAttributes": {
	//       "description": "Flag to include full product data in the results of this request. The default value is false.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "productstatuses/batch",
	//   "request": {
	//     "$ref": "ProductstatusesCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "ProductstatusesCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.productstatuses.get":

type ProductstatusesGetCall struct {
	s            *APIService
	merchantId   uint64
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the status of a product from your Merchant Center account.
// This method can only be called for non-multi-client accounts.
func (r *ProductstatusesService) Get(merchantId uint64, productId string) *ProductstatusesGetCall {
	c := &ProductstatusesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.productId = productId
	return c
}

// IncludeAttributes sets the optional parameter "includeAttributes":
// Flag to include full product data in the result of this get request.
// The default value is false.
func (c *ProductstatusesGetCall) IncludeAttributes(includeAttributes bool) *ProductstatusesGetCall {
	c.urlParams_.Set("includeAttributes", fmt.Sprint(includeAttributes))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductstatusesGetCall) Fields(s ...googleapi.Field) *ProductstatusesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductstatusesGetCall) IfNoneMatch(entityTag string) *ProductstatusesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductstatusesGetCall) Context(ctx context.Context) *ProductstatusesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductstatusesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductstatusesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/productstatuses/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"productId":  c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.productstatuses.get" call.
// Exactly one of *ProductStatus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProductStatus.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductstatusesGetCall) Do(opts ...googleapi.CallOption) (*ProductStatus, error) {
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
	ret := &ProductStatus{
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
	//   "description": "Gets the status of a product from your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.productstatuses.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "includeAttributes": {
	//       "description": "Flag to include full product data in the result of this get request. The default value is false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/productstatuses/{productId}",
	//   "response": {
	//     "$ref": "ProductStatus"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.productstatuses.list":

type ProductstatusesListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the statuses of the products in your Merchant Center
// account. This method can only be called for non-multi-client
// accounts.
func (r *ProductstatusesService) List(merchantId uint64) *ProductstatusesListCall {
	c := &ProductstatusesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// IncludeAttributes sets the optional parameter "includeAttributes":
// Flag to include full product data in the results of the list request.
// The default value is false.
func (c *ProductstatusesListCall) IncludeAttributes(includeAttributes bool) *ProductstatusesListCall {
	c.urlParams_.Set("includeAttributes", fmt.Sprint(includeAttributes))
	return c
}

// IncludeInvalidInsertedItems sets the optional parameter
// "includeInvalidInsertedItems": Flag to include the invalid inserted
// items in the result of the list request. By default the invalid items
// are not shown (the default value is false).
func (c *ProductstatusesListCall) IncludeInvalidInsertedItems(includeInvalidInsertedItems bool) *ProductstatusesListCall {
	c.urlParams_.Set("includeInvalidInsertedItems", fmt.Sprint(includeInvalidInsertedItems))
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of product statuses to return in the response, used for
// paging.
func (c *ProductstatusesListCall) MaxResults(maxResults int64) *ProductstatusesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *ProductstatusesListCall) PageToken(pageToken string) *ProductstatusesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductstatusesListCall) Fields(s ...googleapi.Field) *ProductstatusesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductstatusesListCall) IfNoneMatch(entityTag string) *ProductstatusesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductstatusesListCall) Context(ctx context.Context) *ProductstatusesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductstatusesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductstatusesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/productstatuses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.productstatuses.list" call.
// Exactly one of *ProductstatusesListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ProductstatusesListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductstatusesListCall) Do(opts ...googleapi.CallOption) (*ProductstatusesListResponse, error) {
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
	ret := &ProductstatusesListResponse{
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
	//   "description": "Lists the statuses of the products in your Merchant Center account. This method can only be called for non-multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.productstatuses.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "includeAttributes": {
	//       "description": "Flag to include full product data in the results of the list request. The default value is false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "includeInvalidInsertedItems": {
	//       "description": "Flag to include the invalid inserted items in the result of the list request. By default the invalid items are not shown (the default value is false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of product statuses to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/productstatuses",
	//   "response": {
	//     "$ref": "ProductstatusesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProductstatusesListCall) Pages(ctx context.Context, f func(*ProductstatusesListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.shippingsettings.custombatch":

type ShippingsettingsCustombatchCall struct {
	s                                  *APIService
	shippingsettingscustombatchrequest *ShippingsettingsCustomBatchRequest
	urlParams_                         gensupport.URLParams
	ctx_                               context.Context
	header_                            http.Header
}

// Custombatch: Retrieves and updates the shipping settings of multiple
// accounts in a single request.
func (r *ShippingsettingsService) Custombatch(shippingsettingscustombatchrequest *ShippingsettingsCustomBatchRequest) *ShippingsettingsCustombatchCall {
	c := &ShippingsettingsCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.shippingsettingscustombatchrequest = shippingsettingscustombatchrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *ShippingsettingsCustombatchCall) DryRun(dryRun bool) *ShippingsettingsCustombatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ShippingsettingsCustombatchCall) Fields(s ...googleapi.Field) *ShippingsettingsCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ShippingsettingsCustombatchCall) Context(ctx context.Context) *ShippingsettingsCustombatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ShippingsettingsCustombatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ShippingsettingsCustombatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.shippingsettingscustombatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "shippingsettings/batch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.shippingsettings.custombatch" call.
// Exactly one of *ShippingsettingsCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ShippingsettingsCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ShippingsettingsCustombatchCall) Do(opts ...googleapi.CallOption) (*ShippingsettingsCustomBatchResponse, error) {
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
	ret := &ShippingsettingsCustomBatchResponse{
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
	//   "description": "Retrieves and updates the shipping settings of multiple accounts in a single request.",
	//   "httpMethod": "POST",
	//   "id": "content.shippingsettings.custombatch",
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "shippingsettings/batch",
	//   "request": {
	//     "$ref": "ShippingsettingsCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "ShippingsettingsCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.shippingsettings.get":

type ShippingsettingsGetCall struct {
	s            *APIService
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the shipping settings of the account. This method can
// only be called for accounts to which the managing account has access:
// either the managing account itself for any Merchant Center account,
// or any sub-account when the managing account is a multi-client
// account.
func (r *ShippingsettingsService) Get(merchantId uint64, accountId uint64) *ShippingsettingsGetCall {
	c := &ShippingsettingsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ShippingsettingsGetCall) Fields(s ...googleapi.Field) *ShippingsettingsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ShippingsettingsGetCall) IfNoneMatch(entityTag string) *ShippingsettingsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ShippingsettingsGetCall) Context(ctx context.Context) *ShippingsettingsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ShippingsettingsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ShippingsettingsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/shippingsettings/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.shippingsettings.get" call.
// Exactly one of *ShippingSettings or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ShippingSettings.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ShippingsettingsGetCall) Do(opts ...googleapi.CallOption) (*ShippingSettings, error) {
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
	ret := &ShippingSettings{
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
	//   "description": "Retrieves the shipping settings of the account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "GET",
	//   "id": "content.shippingsettings.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update shipping settings.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/shippingsettings/{accountId}",
	//   "response": {
	//     "$ref": "ShippingSettings"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.shippingsettings.getsupportedcarriers":

type ShippingsettingsGetsupportedcarriersCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Getsupportedcarriers: Retrieves supported carriers and carrier
// services for an account.
func (r *ShippingsettingsService) Getsupportedcarriers(merchantId uint64) *ShippingsettingsGetsupportedcarriersCall {
	c := &ShippingsettingsGetsupportedcarriersCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ShippingsettingsGetsupportedcarriersCall) Fields(s ...googleapi.Field) *ShippingsettingsGetsupportedcarriersCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ShippingsettingsGetsupportedcarriersCall) IfNoneMatch(entityTag string) *ShippingsettingsGetsupportedcarriersCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ShippingsettingsGetsupportedcarriersCall) Context(ctx context.Context) *ShippingsettingsGetsupportedcarriersCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ShippingsettingsGetsupportedcarriersCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ShippingsettingsGetsupportedcarriersCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/supportedCarriers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.shippingsettings.getsupportedcarriers" call.
// Exactly one of *ShippingsettingsGetSupportedCarriersResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *ShippingsettingsGetSupportedCarriersResponse.ServerResponse.Header
// or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ShippingsettingsGetsupportedcarriersCall) Do(opts ...googleapi.CallOption) (*ShippingsettingsGetSupportedCarriersResponse, error) {
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
	ret := &ShippingsettingsGetSupportedCarriersResponse{
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
	//   "description": "Retrieves supported carriers and carrier services for an account.",
	//   "httpMethod": "GET",
	//   "id": "content.shippingsettings.getsupportedcarriers",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "merchantId": {
	//       "description": "The ID of the account for which to retrieve the supported carriers.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/supportedCarriers",
	//   "response": {
	//     "$ref": "ShippingsettingsGetSupportedCarriersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.shippingsettings.list":

type ShippingsettingsListCall struct {
	s            *APIService
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the shipping settings of the sub-accounts in your
// Merchant Center account. This method can only be called for
// multi-client accounts.
func (r *ShippingsettingsService) List(merchantId uint64) *ShippingsettingsListCall {
	c := &ShippingsettingsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of shipping settings to return in the response, used for
// paging.
func (c *ShippingsettingsListCall) MaxResults(maxResults int64) *ShippingsettingsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *ShippingsettingsListCall) PageToken(pageToken string) *ShippingsettingsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ShippingsettingsListCall) Fields(s ...googleapi.Field) *ShippingsettingsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ShippingsettingsListCall) IfNoneMatch(entityTag string) *ShippingsettingsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ShippingsettingsListCall) Context(ctx context.Context) *ShippingsettingsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ShippingsettingsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ShippingsettingsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/shippingsettings")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.shippingsettings.list" call.
// Exactly one of *ShippingsettingsListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ShippingsettingsListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ShippingsettingsListCall) Do(opts ...googleapi.CallOption) (*ShippingsettingsListResponse, error) {
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
	ret := &ShippingsettingsListResponse{
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
	//   "description": "Lists the shipping settings of the sub-accounts in your Merchant Center account. This method can only be called for multi-client accounts.",
	//   "httpMethod": "GET",
	//   "id": "content.shippingsettings.list",
	//   "parameterOrder": [
	//     "merchantId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of shipping settings to return in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/shippingsettings",
	//   "response": {
	//     "$ref": "ShippingsettingsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ShippingsettingsListCall) Pages(ctx context.Context, f func(*ShippingsettingsListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "content.shippingsettings.patch":

type ShippingsettingsPatchCall struct {
	s                *APIService
	merchantId       uint64
	accountId        uint64
	shippingsettings *ShippingSettings
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Patch: Updates the shipping settings of the account. This method can
// only be called for accounts to which the managing account has access:
// either the managing account itself for any Merchant Center account,
// or any sub-account when the managing account is a multi-client
// account. This method supports patch semantics.
func (r *ShippingsettingsService) Patch(merchantId uint64, accountId uint64, shippingsettings *ShippingSettings) *ShippingsettingsPatchCall {
	c := &ShippingsettingsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.shippingsettings = shippingsettings
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *ShippingsettingsPatchCall) DryRun(dryRun bool) *ShippingsettingsPatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ShippingsettingsPatchCall) Fields(s ...googleapi.Field) *ShippingsettingsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ShippingsettingsPatchCall) Context(ctx context.Context) *ShippingsettingsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ShippingsettingsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ShippingsettingsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.shippingsettings)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/shippingsettings/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.shippingsettings.patch" call.
// Exactly one of *ShippingSettings or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ShippingSettings.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ShippingsettingsPatchCall) Do(opts ...googleapi.CallOption) (*ShippingSettings, error) {
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
	ret := &ShippingSettings{
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
	//   "description": "Updates the shipping settings of the account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "content.shippingsettings.patch",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update shipping settings.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/shippingsettings/{accountId}",
	//   "request": {
	//     "$ref": "ShippingSettings"
	//   },
	//   "response": {
	//     "$ref": "ShippingSettings"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.shippingsettings.update":

type ShippingsettingsUpdateCall struct {
	s                *APIService
	merchantId       uint64
	accountId        uint64
	shippingsettings *ShippingSettings
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Update: Updates the shipping settings of the account. This method can
// only be called for accounts to which the managing account has access:
// either the managing account itself for any Merchant Center account,
// or any sub-account when the managing account is a multi-client
// account.
func (r *ShippingsettingsService) Update(merchantId uint64, accountId uint64, shippingsettings *ShippingSettings) *ShippingsettingsUpdateCall {
	c := &ShippingsettingsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.shippingsettings = shippingsettings
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *ShippingsettingsUpdateCall) DryRun(dryRun bool) *ShippingsettingsUpdateCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ShippingsettingsUpdateCall) Fields(s ...googleapi.Field) *ShippingsettingsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ShippingsettingsUpdateCall) Context(ctx context.Context) *ShippingsettingsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ShippingsettingsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ShippingsettingsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.shippingsettings)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/shippingsettings/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "content.shippingsettings.update" call.
// Exactly one of *ShippingSettings or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ShippingSettings.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ShippingsettingsUpdateCall) Do(opts ...googleapi.CallOption) (*ShippingSettings, error) {
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
	ret := &ShippingSettings{
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
	//   "description": "Updates the shipping settings of the account. This method can only be called for accounts to which the managing account has access: either the managing account itself for any Merchant Center account, or any sub-account when the managing account is a multi-client account.",
	//   "httpMethod": "PUT",
	//   "id": "content.shippingsettings.update",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update shipping settings.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "merchantId": {
	//       "description": "The ID of the managing account.",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{merchantId}/shippingsettings/{accountId}",
	//   "request": {
	//     "$ref": "ShippingSettings"
	//   },
	//   "response": {
	//     "$ref": "ShippingSettings"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}
