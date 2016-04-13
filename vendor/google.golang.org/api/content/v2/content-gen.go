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

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Accounts = NewAccountsService(s)
	s.Accountshipping = NewAccountshippingService(s)
	s.Accountstatuses = NewAccountstatusesService(s)
	s.Accounttax = NewAccounttaxService(s)
	s.Datafeeds = NewDatafeedsService(s)
	s.Datafeedstatuses = NewDatafeedstatusesService(s)
	s.Inventory = NewInventoryService(s)
	s.Orders = NewOrdersService(s)
	s.Products = NewProductsService(s)
	s.Productstatuses = NewProductstatusesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Accounts *AccountsService

	Accountshipping *AccountshippingService

	Accountstatuses *AccountstatusesService

	Accounttax *AccounttaxService

	Datafeeds *DatafeedsService

	Datafeedstatuses *DatafeedstatusesService

	Inventory *InventoryService

	Orders *OrdersService

	Products *ProductsService

	Productstatuses *ProductstatusesService
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

func NewAccountshippingService(s *Service) *AccountshippingService {
	rs := &AccountshippingService{s: s}
	return rs
}

type AccountshippingService struct {
	s *Service
}

func NewAccountstatusesService(s *Service) *AccountstatusesService {
	rs := &AccountstatusesService{s: s}
	return rs
}

type AccountstatusesService struct {
	s *Service
}

func NewAccounttaxService(s *Service) *AccounttaxService {
	rs := &AccounttaxService{s: s}
	return rs
}

type AccounttaxService struct {
	s *Service
}

func NewDatafeedsService(s *Service) *DatafeedsService {
	rs := &DatafeedsService{s: s}
	return rs
}

type DatafeedsService struct {
	s *Service
}

func NewDatafeedstatusesService(s *Service) *DatafeedstatusesService {
	rs := &DatafeedstatusesService{s: s}
	return rs
}

type DatafeedstatusesService struct {
	s *Service
}

func NewInventoryService(s *Service) *InventoryService {
	rs := &InventoryService{s: s}
	return rs
}

type InventoryService struct {
	s *Service
}

func NewOrdersService(s *Service) *OrdersService {
	rs := &OrdersService{s: s}
	return rs
}

type OrdersService struct {
	s *Service
}

func NewProductsService(s *Service) *ProductsService {
	rs := &ProductsService{s: s}
	return rs
}

type ProductsService struct {
	s *Service
}

func NewProductstatusesService(s *Service) *ProductstatusesService {
	rs := &ProductstatusesService{s: s}
	return rs
}

type ProductstatusesService struct {
	s *Service
}

// Account: Account data.
type Account struct {
	// AdultContent: Indicates whether the merchant sells adult content.
	AdultContent bool `json:"adultContent,omitempty"`

	// AdwordsLinks: List of linked AdWords accounts, active or pending
	// approval. To create a new link request, add a new link with status
	// active to the list. It will remain is state pending until approved or
	// rejected in the AdWords interface. To delete an active link or to
	// cancel a link request, remove it from the list.
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
}

func (s *Account) MarshalJSON() ([]byte, error) {
	type noMethod Account
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountAdwordsLink) MarshalJSON() ([]byte, error) {
	type noMethod AccountAdwordsLink
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountIdentifier) MarshalJSON() ([]byte, error) {
	type noMethod AccountIdentifier
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShipping: The shipping settings of a merchant account.
type AccountShipping struct {
	// AccountId: The ID of the account to which these account shipping
	// settings belong.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// CarrierRates: Carrier-based shipping calculations.
	CarrierRates []*AccountShippingCarrierRate `json:"carrierRates,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountShipping".
	Kind string `json:"kind,omitempty"`

	// LocationGroups: Location groups for shipping.
	LocationGroups []*AccountShippingLocationGroup `json:"locationGroups,omitempty"`

	// RateTables: Rate tables definitions.
	RateTables []*AccountShippingRateTable `json:"rateTables,omitempty"`

	// Services: Shipping services describing shipping fees calculation.
	Services []*AccountShippingShippingService `json:"services,omitempty"`

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

func (s *AccountShipping) MarshalJSON() ([]byte, error) {
	type noMethod AccountShipping
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShippingCarrierRate: A carrier-calculated shipping rate.
type AccountShippingCarrierRate struct {
	// Carrier: The carrier that is responsible for the shipping, such as
	// "UPS", "FedEx", or "USPS".
	Carrier string `json:"carrier,omitempty"`

	// CarrierService: The carrier service, such as "Ground" or "2Day".
	CarrierService string `json:"carrierService,omitempty"`

	// ModifierFlatRate: Additive shipping rate modifier.
	ModifierFlatRate *Price `json:"modifierFlatRate,omitempty"`

	// ModifierPercent: Multiplicative shipping rate modifier in percent.
	// Represented as a floating point number without the percentage
	// character.
	ModifierPercent string `json:"modifierPercent,omitempty"`

	// Name: The name of the carrier rate.
	Name string `json:"name,omitempty"`

	// SaleCountry: The sale country for which this carrier rate is valid,
	// represented as a CLDR territory code.
	SaleCountry string `json:"saleCountry,omitempty"`

	// ShippingOrigin: Shipping origin represented as a postal code.
	ShippingOrigin string `json:"shippingOrigin,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingCarrierRate) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingCarrierRate
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AccountShippingCondition struct {
	// DeliveryLocationGroup: Delivery location in terms of a location group
	// name. A location group with this name must be specified among
	// location groups.
	DeliveryLocationGroup string `json:"deliveryLocationGroup,omitempty"`

	// DeliveryLocationId: Delivery location in terms of a location ID. Can
	// be used to represent administrative areas, smaller country
	// subdivisions, or cities.
	DeliveryLocationId int64 `json:"deliveryLocationId,omitempty,string"`

	// DeliveryPostalCode: Delivery location in terms of a postal code.
	DeliveryPostalCode string `json:"deliveryPostalCode,omitempty"`

	// DeliveryPostalCodeRange: Delivery location in terms of a postal code
	// range.
	DeliveryPostalCodeRange *AccountShippingPostalCodeRange `json:"deliveryPostalCodeRange,omitempty"`

	// PriceMax: Maximum shipping price. Forms an interval between the
	// maximum of smaller prices (exclusive) and this price (inclusive).
	PriceMax *Price `json:"priceMax,omitempty"`

	// ShippingLabel: Shipping label of the product. The products with the
	// label are matched.
	ShippingLabel string `json:"shippingLabel,omitempty"`

	// WeightMax: Maximum shipping weight. Forms an interval between the
	// maximum of smaller weight (exclusive) and this weight (inclusive).
	WeightMax *Weight `json:"weightMax,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DeliveryLocationGroup") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingCondition) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingCondition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShippingLocationGroup: A user-defined locations group in a
// given country. All the locations of the group must be of the same
// type.
type AccountShippingLocationGroup struct {
	// Country: The CLDR territory code of the country in which this
	// location group is.
	Country string `json:"country,omitempty"`

	// LocationIds: A location ID (also called criteria ID) representing
	// administrative areas, smaller country subdivisions (counties), or
	// cities.
	LocationIds googleapi.Int64s `json:"locationIds,omitempty"`

	// Name: The name of the location group.
	Name string `json:"name,omitempty"`

	// PostalCodeRanges: A postal code range representing a city or a set of
	// cities.
	PostalCodeRanges []*AccountShippingPostalCodeRange `json:"postalCodeRanges,omitempty"`

	// PostalCodes: A postal code representing a city or a set of cities.
	//
	// - A single postal code (e.g., 12345)
	// - A postal code prefix followed by a star (e.g., 1234*)
	PostalCodes []string `json:"postalCodes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingLocationGroup) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingLocationGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShippingPostalCodeRange: A postal code range, that can be
// either:
// - A range of postal codes (e.g., start=12340, end=12359)
// - A range of postal codes prefixes (e.g., start=1234* end=1235*).
// Prefixes must be of the same length (e.g., start=12* end=2* is
// invalid).
type AccountShippingPostalCodeRange struct {
	// End: The last (inclusive) postal code or prefix of the range.
	End string `json:"end,omitempty"`

	// Start: The first (inclusive) postal code or prefix of the range.
	Start string `json:"start,omitempty"`

	// ForceSendFields is a list of field names (e.g. "End") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingPostalCodeRange) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingPostalCodeRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShippingRateTable: A single or bi-dimensional table of
// shipping rates. Each dimension is defined in terms of consecutive
// price/weight ranges, delivery locations, or shipping labels.
type AccountShippingRateTable struct {
	// Content: One-dimensional table cells define one condition along the
	// same dimension. Bi-dimensional table cells use two dimensions with
	// respectively M and N distinct values and must contain exactly M * N
	// cells with distinct conditions (for each possible value pairs).
	Content []*AccountShippingRateTableCell `json:"content,omitempty"`

	// Name: The name of the rate table.
	Name string `json:"name,omitempty"`

	// SaleCountry: The sale country for which this table is valid,
	// represented as a CLDR territory code.
	SaleCountry string `json:"saleCountry,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Content") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingRateTable) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingRateTable
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AccountShippingRateTableCell struct {
	// Condition: Conditions for which the cell is valid. All cells in a
	// table must use the same dimension or pair of dimensions among price,
	// weight, shipping label or delivery location. If no condition is
	// specified, the cell acts as a catch-all and matches all the elements
	// that are not matched by other cells in this dimension.
	Condition *AccountShippingCondition `json:"condition,omitempty"`

	// Rate: The rate applicable if the cell conditions are matched.
	Rate *Price `json:"rate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Condition") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingRateTableCell) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingRateTableCell
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShippingShippingService: Shipping services provided in a
// country.
type AccountShippingShippingService struct {
	// Active: Whether the shipping service is available.
	Active bool `json:"active,omitempty"`

	// CalculationMethod: Calculation method for the "simple" case that
	// needs no rules.
	CalculationMethod *AccountShippingShippingServiceCalculationMethod `json:"calculationMethod,omitempty"`

	// CostRuleTree: Decision tree for "complicated" shipping cost
	// calculation.
	CostRuleTree *AccountShippingShippingServiceCostRule `json:"costRuleTree,omitempty"`

	// Name: The name of this shipping service.
	Name string `json:"name,omitempty"`

	// SaleCountry: The CLDR territory code of the sale country for which
	// this service can be used.
	SaleCountry string `json:"saleCountry,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Active") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingShippingService) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingShippingService
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShippingShippingServiceCalculationMethod: Shipping cost
// calculation method. Exactly one of the field is set.
type AccountShippingShippingServiceCalculationMethod struct {
	// CarrierRate: Name of the carrier rate to use for the calculation.
	CarrierRate string `json:"carrierRate,omitempty"`

	// Excluded: Delivery is excluded. Valid only within cost rules tree.
	Excluded bool `json:"excluded,omitempty"`

	// FlatRate: Fixed price shipping, represented as a floating point
	// number associated with a currency.
	FlatRate *Price `json:"flatRate,omitempty"`

	// PercentageRate: Percentage of the price, represented as a floating
	// point number without the percentage character.
	PercentageRate string `json:"percentageRate,omitempty"`

	// RateTable: Name of the rate table to use for the calculation.
	RateTable string `json:"rateTable,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CarrierRate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingShippingServiceCalculationMethod) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingShippingServiceCalculationMethod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountShippingShippingServiceCostRule: Building block of the cost
// calculation decision tree.
// - The tree root should have no condition and no calculation method.
// - All the children must have a condition on the same dimension. The
// first child matching a condition is entered, therefore, price and
// weight conditions form contiguous intervals.
// - The last child of an element must have no condition and matches all
// elements not previously matched.
// - Children and calculation method are mutually exclusive, and exactly
// one of them must be defined; the root must only have children.
type AccountShippingShippingServiceCostRule struct {
	// CalculationMethod: Final calculation method to be used only in leaf
	// nodes.
	CalculationMethod *AccountShippingShippingServiceCalculationMethod `json:"calculationMethod,omitempty"`

	// Children: Subsequent rules to be applied, only for inner nodes. The
	// last child must not specify a condition and acts as a catch-all.
	Children []*AccountShippingShippingServiceCostRule `json:"children,omitempty"`

	// Condition: Condition for this rule to be applicable. If no condition
	// is specified, the rule acts as a catch-all.
	Condition *AccountShippingCondition `json:"condition,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CalculationMethod")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountShippingShippingServiceCostRule) MarshalJSON() ([]byte, error) {
	type noMethod AccountShippingShippingServiceCostRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountStatus: The status of an account, i.e., information about its
// products, which is computed offline and not returned immediately at
// insertion time.
type AccountStatus struct {
	// AccountId: The ID of the account for which the status is reported.
	AccountId string `json:"accountId,omitempty"`

	// DataQualityIssues: A list of data quality issues.
	DataQualityIssues []*AccountStatusDataQualityIssue `json:"dataQualityIssues,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountStatus".
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

func (s *AccountStatus) MarshalJSON() ([]byte, error) {
	type noMethod AccountStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AccountStatusDataQualityIssue struct {
	// Country: Country for which this issue is reported.
	Country string `json:"country,omitempty"`

	// DisplayedValue: Actual value displayed on the landing page.
	DisplayedValue string `json:"displayedValue,omitempty"`

	// ExampleItems: Example items featuring the issue.
	ExampleItems []*AccountStatusExampleItem `json:"exampleItems,omitempty"`

	// Id: Issue identifier.
	Id string `json:"id,omitempty"`

	// LastChecked: Last time the account was checked for this issue.
	LastChecked string `json:"lastChecked,omitempty"`

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
}

func (s *AccountStatusDataQualityIssue) MarshalJSON() ([]byte, error) {
	type noMethod AccountStatusDataQualityIssue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountStatusExampleItem) MarshalJSON() ([]byte, error) {
	type noMethod AccountStatusExampleItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountTax) MarshalJSON() ([]byte, error) {
	type noMethod AccountTax
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountTaxTaxRule) MarshalJSON() ([]byte, error) {
	type noMethod AccountTaxTaxRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountUser) MarshalJSON() ([]byte, error) {
	type noMethod AccountUser
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountsAuthInfoResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountsAuthInfoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountsCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountsCustomBatchRequestEntry: A batch entry encoding a single
// non-batch accounts request.
type AccountsCustomBatchRequestEntry struct {
	// Account: The account to create or update. Only defined if the method
	// is insert or update.
	Account *Account `json:"account,omitempty"`

	// AccountId: The ID of the account to get or delete. Only defined if
	// the method is get or delete.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

	// MerchantId: The ID of the managing account.
	MerchantId uint64 `json:"merchantId,omitempty,string"`

	Method string `json:"method,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Account") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountsCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountsCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountsCustomBatchResponseEntry: A batch entry encoding a single
// non-batch accounts response.
type AccountsCustomBatchResponseEntry struct {
	// Account: The retrieved, created, or updated account. Not defined if
	// the method was delete.
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
}

func (s *AccountsCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountsCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AccountshippingCustomBatchRequest struct {
	// Entries: The request entries to be processed in the batch.
	Entries []*AccountshippingCustomBatchRequestEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountshippingCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod AccountshippingCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountshippingCustomBatchRequestEntry: A batch entry encoding a
// single non-batch accountshipping request.
type AccountshippingCustomBatchRequestEntry struct {
	// AccountId: The ID of the account for which to get/update account
	// shipping settings.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// AccountShipping: The account shipping settings to update. Only
	// defined if the method is update.
	AccountShipping *AccountShipping `json:"accountShipping,omitempty"`

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
}

func (s *AccountshippingCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountshippingCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AccountshippingCustomBatchResponse struct {
	// Entries: The result of the execution of the batch requests.
	Entries []*AccountshippingCustomBatchResponseEntry `json:"entries,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountshippingCustomBatchResponse".
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
}

func (s *AccountshippingCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountshippingCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AccountshippingCustomBatchResponseEntry: A batch entry encoding a
// single non-batch accountshipping response.
type AccountshippingCustomBatchResponseEntry struct {
	// AccountShipping: The retrieved or updated account shipping settings.
	AccountShipping *AccountShipping `json:"accountShipping,omitempty"`

	// BatchId: The ID of the request entry this entry responds to.
	BatchId int64 `json:"batchId,omitempty"`

	// Errors: A list of errors defined if and only if the request failed.
	Errors *Errors `json:"errors,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountshippingCustomBatchResponseEntry".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountShipping") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccountshippingCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountshippingCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AccountshippingListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "content#accountshippingListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token for the retrieval of the next page of
	// account shipping settings.
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*AccountShipping `json:"resources,omitempty"`

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

func (s *AccountshippingListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountshippingListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountstatusesCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountstatusesCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountstatusesCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountstatusesCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccountstatusesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccountstatusesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccounttaxCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccounttaxCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccounttaxCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccounttaxCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AccounttaxListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AccounttaxListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Datafeed: Datafeed data.
type Datafeed struct {
	// AttributeLanguage: The two-letter ISO 639-1 language in which the
	// attributes are defined in the data feed.
	AttributeLanguage string `json:"attributeLanguage,omitempty"`

	// ContentLanguage: The two-letter ISO 639-1 language of the items in
	// the feed. Must be a valid language for targetCountry.
	ContentLanguage string `json:"contentLanguage,omitempty"`

	// ContentType: The type of data feed.
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
}

func (s *Datafeed) MarshalJSON() ([]byte, error) {
	type noMethod Datafeed
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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

	// Hour: The hour of the day the feed file should be fetched (0-24).
	Hour int64 `json:"hour,omitempty"`

	// Password: An optional password for fetch_url.
	Password string `json:"password,omitempty"`

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
}

func (s *DatafeedFetchSchedule) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedFetchSchedule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedFormat) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedFormat
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedStatus) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedStatusError) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedStatusError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedStatusExample) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedStatusExample
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedsCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedsCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedsCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedsCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedstatusesCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedstatusesCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedstatusesCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedstatusesCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DatafeedstatusesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatafeedstatusesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *Error) MarshalJSON() ([]byte, error) {
	type noMethod Error
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *Errors) MarshalJSON() ([]byte, error) {
	type noMethod Errors
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *Installment) MarshalJSON() ([]byte, error) {
	type noMethod Installment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *Inventory) MarshalJSON() ([]byte, error) {
	type noMethod Inventory
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *InventoryCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *InventoryCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *InventoryCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *InventoryCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod InventoryCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *InventorySetRequest) MarshalJSON() ([]byte, error) {
	type noMethod InventorySetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *InventorySetResponse) MarshalJSON() ([]byte, error) {
	type noMethod InventorySetResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *LoyaltyPoints) MarshalJSON() ([]byte, error) {
	type noMethod LoyaltyPoints
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Order struct {
	// Acknowledged: Whether the order was acknowledged.
	Acknowledged bool `json:"acknowledged,omitempty"`

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
}

func (s *Order) MarshalJSON() ([]byte, error) {
	type noMethod Order
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderAddress) MarshalJSON() ([]byte, error) {
	type noMethod OrderAddress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OrderCancellation struct {
	// Actor: The actor that created the cancellation.
	Actor string `json:"actor,omitempty"`

	// CreationDate: Date on which the cancellation has been created, in ISO
	// 8601 format.
	CreationDate string `json:"creationDate,omitempty"`

	// Quantity: The quantity that was canceled.
	Quantity int64 `json:"quantity,omitempty"`

	// Reason: The reason for the cancellation.
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
}

func (s *OrderCancellation) MarshalJSON() ([]byte, error) {
	type noMethod OrderCancellation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OrderCustomer struct {
	// Email: Email address of the customer.
	Email string `json:"email,omitempty"`

	// ExplicitMarketingPreference: If set, this indicates the user had a
	// choice to opt in or out of providing marketing rights to the
	// merchant. If unset, this indicates the user has already made this
	// choice in a previous purchase, and was thus not shown the marketing
	// right opt in/out checkbox during the Purchases on Google checkout
	// flow.
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
}

func (s *OrderCustomer) MarshalJSON() ([]byte, error) {
	type noMethod OrderCustomer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderDeliveryDetails) MarshalJSON() ([]byte, error) {
	type noMethod OrderDeliveryDetails
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderLineItem) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderLineItemProduct) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemProduct
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderLineItemProductVariantAttribute) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemProductVariantAttribute
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderLineItemReturnInfo) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemReturnInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderLineItemShippingDetails) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemShippingDetails
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderLineItemShippingDetailsMethod) MarshalJSON() ([]byte, error) {
	type noMethod OrderLineItemShippingDetailsMethod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderPaymentMethod) MarshalJSON() ([]byte, error) {
	type noMethod OrderPaymentMethod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderRefund) MarshalJSON() ([]byte, error) {
	type noMethod OrderRefund
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderReturn) MarshalJSON() ([]byte, error) {
	type noMethod OrderReturn
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderShipment) MarshalJSON() ([]byte, error) {
	type noMethod OrderShipment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrderShipmentLineItemShipment) MarshalJSON() ([]byte, error) {
	type noMethod OrderShipmentLineItemShipment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersAcknowledgeRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersAcknowledgeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersAcknowledgeResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersAcknowledgeResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersAdvanceTestOrderResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersAdvanceTestOrderResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCancelLineItemRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelLineItemRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCancelLineItemResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelLineItemResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCancelRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCancelResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCancelResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCreateTestOrderRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCreateTestOrderRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCreateTestOrderResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCreateTestOrderResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequestEntryCancel) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryCancel
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequestEntryCancelLineItem) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryCancelLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequestEntryRefund) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryRefund
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequestEntryReturnLineItem) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryReturnLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequestEntryShipLineItems) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryShipLineItems
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchRequestEntryUpdateShipment) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchRequestEntryUpdateShipment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod OrdersCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersGetByMerchantOrderIdResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersGetByMerchantOrderIdResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersGetTestOrderTemplateResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersGetTestOrderTemplateResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersRefundRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersRefundRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersRefundResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersRefundResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersReturnLineItemRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersReturnLineItemRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersReturnLineItemResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersReturnLineItemResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersShipLineItemsRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersShipLineItemsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersShipLineItemsResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersShipLineItemsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersUpdateMerchantOrderIdRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateMerchantOrderIdRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersUpdateMerchantOrderIdResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateMerchantOrderIdResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersUpdateShipmentRequest) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateShipmentRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *OrdersUpdateShipmentResponse) MarshalJSON() ([]byte, error) {
	type noMethod OrdersUpdateShipmentResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *Price) MarshalJSON() ([]byte, error) {
	type noMethod Price
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Product: Product data.
type Product struct {
	// AdditionalImageLinks: Additional URLs of images of the item.
	AdditionalImageLinks []string `json:"additionalImageLinks,omitempty"`

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

	// MobileLink: Link to a mobile-optimized version of the landing page.
	MobileLink string `json:"mobileLink,omitempty"`

	// Mpn: Manufacturer Part Number (MPN) of the item.
	Mpn string `json:"mpn,omitempty"`

	// Multipack: The number of identical products in a merchant-defined
	// multipack.
	Multipack int64 `json:"multipack,omitempty,string"`

	// OfferId: An identifier of the item.
	OfferId string `json:"offerId,omitempty"`

	// OnlineOnly: Whether an item is available for purchase only online.
	OnlineOnly bool `json:"onlineOnly,omitempty"`

	// Pattern: The item's pattern (e.g. polka dots).
	Pattern string `json:"pattern,omitempty"`

	// Price: Price of the item.
	Price *Price `json:"price,omitempty"`

	// ProductType: Your category of the item (formatted as in product feeds
	// specification).
	ProductType string `json:"productType,omitempty"`

	// SalePrice: Advertised sale price of the item.
	SalePrice *Price `json:"salePrice,omitempty"`

	// SalePriceEffectiveDate: Date range during which the item is on sale
	// (see product feed specifications).
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
}

func (s *Product) MarshalJSON() ([]byte, error) {
	type noMethod Product
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductAspect) MarshalJSON() ([]byte, error) {
	type noMethod ProductAspect
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductCustomAttribute) MarshalJSON() ([]byte, error) {
	type noMethod ProductCustomAttribute
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductCustomGroup) MarshalJSON() ([]byte, error) {
	type noMethod ProductCustomGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductDestination) MarshalJSON() ([]byte, error) {
	type noMethod ProductDestination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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

	// Region: The geographic region to which a shipping rate applies (e.g.
	// zip code).
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
}

func (s *ProductShipping) MarshalJSON() ([]byte, error) {
	type noMethod ProductShipping
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductShippingDimension) MarshalJSON() ([]byte, error) {
	type noMethod ProductShippingDimension
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductShippingWeight) MarshalJSON() ([]byte, error) {
	type noMethod ProductShippingWeight
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductStatus) MarshalJSON() ([]byte, error) {
	type noMethod ProductStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductStatusDataQualityIssue) MarshalJSON() ([]byte, error) {
	type noMethod ProductStatusDataQualityIssue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductStatusDestinationStatus) MarshalJSON() ([]byte, error) {
	type noMethod ProductStatusDestinationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductTax) MarshalJSON() ([]byte, error) {
	type noMethod ProductTax
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductUnitPricingBaseMeasure) MarshalJSON() ([]byte, error) {
	type noMethod ProductUnitPricingBaseMeasure
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductUnitPricingMeasure) MarshalJSON() ([]byte, error) {
	type noMethod ProductUnitPricingMeasure
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductsCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductsCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductsCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductsCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductsCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductstatusesCustomBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ProductstatusesCustomBatchRequestEntry: A batch entry encoding a
// single non-batch productstatuses request.
type ProductstatusesCustomBatchRequestEntry struct {
	// BatchId: An entry ID, unique within the batch request.
	BatchId int64 `json:"batchId,omitempty"`

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
}

func (s *ProductstatusesCustomBatchRequestEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchRequestEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductstatusesCustomBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductstatusesCustomBatchResponseEntry) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesCustomBatchResponseEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductstatusesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductstatusesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *TestOrder) MarshalJSON() ([]byte, error) {
	type noMethod TestOrder
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TestOrderCustomer struct {
	// Email: Email address of the customer.
	Email string `json:"email,omitempty"`

	// ExplicitMarketingPreference: If set, this indicates the user had a
	// choice to opt in or out of providing marketing rights to the
	// merchant. If unset, this indicates the user has already made this
	// choice in a previous purchase, and was thus not shown the marketing
	// right opt in/out checkbox during the Purchases on Google checkout
	// flow. Optional.
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
}

func (s *TestOrderCustomer) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderCustomer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *TestOrderLineItem) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderLineItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *TestOrderLineItemProduct) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderLineItemProduct
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *TestOrderPaymentMethod) MarshalJSON() ([]byte, error) {
	type noMethod TestOrderPaymentMethod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *Weight) MarshalJSON() ([]byte, error) {
	type noMethod Weight
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "content.accounts.authinfo":

type AccountsAuthinfoCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Authinfo: Returns information about the authenticated user.
func (r *AccountsService) Authinfo() *AccountsAuthinfoCall {
	c := &AccountsAuthinfoCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsAuthinfoCall) QuotaUser(quotaUser string) *AccountsAuthinfoCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsAuthinfoCall) UserIP(userIP string) *AccountsAuthinfoCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccountsAuthinfoCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/authinfo")
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

// Do executes the "content.accounts.authinfo" call.
// Exactly one of *AccountsAuthInfoResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AccountsAuthInfoResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsAuthinfoCall) Do() (*AccountsAuthInfoResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// method id "content.accounts.custombatch":

type AccountsCustombatchCall struct {
	s                          *Service
	accountscustombatchrequest *AccountsCustomBatchRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsCustombatchCall) QuotaUser(quotaUser string) *AccountsCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsCustombatchCall) UserIP(userIP string) *AccountsCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccountsCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accountscustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/batch")
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

// Do executes the "content.accounts.custombatch" call.
// Exactly one of *AccountsCustomBatchResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AccountsCustomBatchResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsCustombatchCall) Do() (*AccountsCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s          *Service
	merchantId uint64
	accountId  uint64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a Merchant Center sub-account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsDeleteCall) QuotaUser(quotaUser string) *AccountsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsDeleteCall) UserIP(userIP string) *AccountsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccountsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accounts.delete" call.
func (c *AccountsDeleteCall) Do() error {
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
	//   "description": "Deletes a Merchant Center sub-account.",
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
	s            *Service
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves a Merchant Center account.
func (r *AccountsService) Get(merchantId uint64, accountId uint64) *AccountsGetCall {
	c := &AccountsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
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

// Do executes the "content.accounts.get" call.
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
	//   "description": "Retrieves a Merchant Center account.",
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
	s          *Service
	merchantId uint64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Creates a Merchant Center sub-account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountsInsertCall) QuotaUser(quotaUser string) *AccountsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountsInsertCall) UserIP(userIP string) *AccountsInsertCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccountsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accounts.insert" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsInsertCall) Do() (*Account, error) {
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
	//   "description": "Creates a Merchant Center sub-account.",
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the sub-accounts in your Merchant Center account.
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.accounts.list" call.
// Exactly one of *AccountsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AccountsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsListCall) Do() (*AccountsListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the sub-accounts in your Merchant Center account.",
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

// method id "content.accounts.patch":

type AccountsPatchCall struct {
	s          *Service
	merchantId uint64
	accountId  uint64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates a Merchant Center account. This method supports patch
// semantics.
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accounts.patch" call.
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
	//   "description": "Updates a Merchant Center account. This method supports patch semantics.",
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
	s          *Service
	merchantId uint64
	accountId  uint64
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates a Merchant Center account.
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounts/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accounts.update" call.
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
	//   "description": "Updates a Merchant Center account.",
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

// method id "content.accountshipping.custombatch":

type AccountshippingCustombatchCall struct {
	s                                 *Service
	accountshippingcustombatchrequest *AccountshippingCustomBatchRequest
	urlParams_                        gensupport.URLParams
	ctx_                              context.Context
}

// Custombatch: Retrieves and updates the shipping settings of multiple
// accounts in a single request.
func (r *AccountshippingService) Custombatch(accountshippingcustombatchrequest *AccountshippingCustomBatchRequest) *AccountshippingCustombatchCall {
	c := &AccountshippingCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountshippingcustombatchrequest = accountshippingcustombatchrequest
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountshippingCustombatchCall) DryRun(dryRun bool) *AccountshippingCustombatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountshippingCustombatchCall) QuotaUser(quotaUser string) *AccountshippingCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountshippingCustombatchCall) UserIP(userIP string) *AccountshippingCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountshippingCustombatchCall) Fields(s ...googleapi.Field) *AccountshippingCustombatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountshippingCustombatchCall) Context(ctx context.Context) *AccountshippingCustombatchCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountshippingCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accountshippingcustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accountshipping/batch")
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

// Do executes the "content.accountshipping.custombatch" call.
// Exactly one of *AccountshippingCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *AccountshippingCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountshippingCustombatchCall) Do() (*AccountshippingCustomBatchResponse, error) {
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
	ret := &AccountshippingCustomBatchResponse{
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
	//   "description": "Retrieves and updates the shipping settings of multiple accounts in a single request.",
	//   "httpMethod": "POST",
	//   "id": "content.accountshipping.custombatch",
	//   "parameters": {
	//     "dryRun": {
	//       "description": "Flag to run the request in dry-run mode.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "accountshipping/batch",
	//   "request": {
	//     "$ref": "AccountshippingCustomBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "AccountshippingCustomBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountshipping.get":

type AccountshippingGetCall struct {
	s            *Service
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the shipping settings of the account.
func (r *AccountshippingService) Get(merchantId uint64, accountId uint64) *AccountshippingGetCall {
	c := &AccountshippingGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountshippingGetCall) QuotaUser(quotaUser string) *AccountshippingGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountshippingGetCall) UserIP(userIP string) *AccountshippingGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountshippingGetCall) Fields(s ...googleapi.Field) *AccountshippingGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountshippingGetCall) IfNoneMatch(entityTag string) *AccountshippingGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountshippingGetCall) Context(ctx context.Context) *AccountshippingGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountshippingGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountshipping/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
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

// Do executes the "content.accountshipping.get" call.
// Exactly one of *AccountShipping or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountShipping.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountshippingGetCall) Do() (*AccountShipping, error) {
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
	ret := &AccountShipping{
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
	//   "description": "Retrieves the shipping settings of the account.",
	//   "httpMethod": "GET",
	//   "id": "content.accountshipping.get",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update account shipping settings.",
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
	//   "path": "{merchantId}/accountshipping/{accountId}",
	//   "response": {
	//     "$ref": "AccountShipping"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountshipping.list":

type AccountshippingListCall struct {
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the shipping settings of the sub-accounts in your
// Merchant Center account.
func (r *AccountshippingService) List(merchantId uint64) *AccountshippingListCall {
	c := &AccountshippingListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of shipping settings to return in the response, used for
// paging.
func (c *AccountshippingListCall) MaxResults(maxResults int64) *AccountshippingListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *AccountshippingListCall) PageToken(pageToken string) *AccountshippingListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountshippingListCall) QuotaUser(quotaUser string) *AccountshippingListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountshippingListCall) UserIP(userIP string) *AccountshippingListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountshippingListCall) Fields(s ...googleapi.Field) *AccountshippingListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountshippingListCall) IfNoneMatch(entityTag string) *AccountshippingListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountshippingListCall) Context(ctx context.Context) *AccountshippingListCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountshippingListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountshipping")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.accountshipping.list" call.
// Exactly one of *AccountshippingListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AccountshippingListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountshippingListCall) Do() (*AccountshippingListResponse, error) {
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
	ret := &AccountshippingListResponse{
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
	//   "description": "Lists the shipping settings of the sub-accounts in your Merchant Center account.",
	//   "httpMethod": "GET",
	//   "id": "content.accountshipping.list",
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
	//   "path": "{merchantId}/accountshipping",
	//   "response": {
	//     "$ref": "AccountshippingListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountshipping.patch":

type AccountshippingPatchCall struct {
	s               *Service
	merchantId      uint64
	accountId       uint64
	accountshipping *AccountShipping
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// Patch: Updates the shipping settings of the account. This method
// supports patch semantics.
func (r *AccountshippingService) Patch(merchantId uint64, accountId uint64, accountshipping *AccountShipping) *AccountshippingPatchCall {
	c := &AccountshippingPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.accountshipping = accountshipping
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountshippingPatchCall) DryRun(dryRun bool) *AccountshippingPatchCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountshippingPatchCall) QuotaUser(quotaUser string) *AccountshippingPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountshippingPatchCall) UserIP(userIP string) *AccountshippingPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountshippingPatchCall) Fields(s ...googleapi.Field) *AccountshippingPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountshippingPatchCall) Context(ctx context.Context) *AccountshippingPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountshippingPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accountshipping)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountshipping/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accountshipping.patch" call.
// Exactly one of *AccountShipping or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountShipping.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountshippingPatchCall) Do() (*AccountShipping, error) {
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
	ret := &AccountShipping{
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
	//   "description": "Updates the shipping settings of the account. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "content.accountshipping.patch",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update account shipping settings.",
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
	//   "path": "{merchantId}/accountshipping/{accountId}",
	//   "request": {
	//     "$ref": "AccountShipping"
	//   },
	//   "response": {
	//     "$ref": "AccountShipping"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountshipping.update":

type AccountshippingUpdateCall struct {
	s               *Service
	merchantId      uint64
	accountId       uint64
	accountshipping *AccountShipping
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// Update: Updates the shipping settings of the account.
func (r *AccountshippingService) Update(merchantId uint64, accountId uint64, accountshipping *AccountShipping) *AccountshippingUpdateCall {
	c := &AccountshippingUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	c.accountshipping = accountshipping
	return c
}

// DryRun sets the optional parameter "dryRun": Flag to run the request
// in dry-run mode.
func (c *AccountshippingUpdateCall) DryRun(dryRun bool) *AccountshippingUpdateCall {
	c.urlParams_.Set("dryRun", fmt.Sprint(dryRun))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountshippingUpdateCall) QuotaUser(quotaUser string) *AccountshippingUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountshippingUpdateCall) UserIP(userIP string) *AccountshippingUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountshippingUpdateCall) Fields(s ...googleapi.Field) *AccountshippingUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountshippingUpdateCall) Context(ctx context.Context) *AccountshippingUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *AccountshippingUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accountshipping)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountshipping/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accountshipping.update" call.
// Exactly one of *AccountShipping or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountShipping.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountshippingUpdateCall) Do() (*AccountShipping, error) {
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
	ret := &AccountShipping{
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
	//   "description": "Updates the shipping settings of the account.",
	//   "httpMethod": "PUT",
	//   "id": "content.accountshipping.update",
	//   "parameterOrder": [
	//     "merchantId",
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The ID of the account for which to get/update account shipping settings.",
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
	//   "path": "{merchantId}/accountshipping/{accountId}",
	//   "request": {
	//     "$ref": "AccountShipping"
	//   },
	//   "response": {
	//     "$ref": "AccountShipping"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/content"
	//   ]
	// }

}

// method id "content.accountstatuses.custombatch":

type AccountstatusesCustombatchCall struct {
	s                                 *Service
	accountstatusescustombatchrequest *AccountstatusesCustomBatchRequest
	urlParams_                        gensupport.URLParams
	ctx_                              context.Context
}

// Custombatch:
func (r *AccountstatusesService) Custombatch(accountstatusescustombatchrequest *AccountstatusesCustomBatchRequest) *AccountstatusesCustombatchCall {
	c := &AccountstatusesCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountstatusescustombatchrequest = accountstatusescustombatchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountstatusesCustombatchCall) QuotaUser(quotaUser string) *AccountstatusesCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountstatusesCustombatchCall) UserIP(userIP string) *AccountstatusesCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccountstatusesCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accountstatusescustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accountstatuses/batch")
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

// Do executes the "content.accountstatuses.custombatch" call.
// Exactly one of *AccountstatusesCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *AccountstatusesCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountstatusesCustombatchCall) Do() (*AccountstatusesCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s            *Service
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the status of a Merchant Center account.
func (r *AccountstatusesService) Get(merchantId uint64, accountId uint64) *AccountstatusesGetCall {
	c := &AccountstatusesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountstatusesGetCall) QuotaUser(quotaUser string) *AccountstatusesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountstatusesGetCall) UserIP(userIP string) *AccountstatusesGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccountstatusesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountstatuses/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
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

// Do executes the "content.accountstatuses.get" call.
// Exactly one of *AccountStatus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountStatus.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountstatusesGetCall) Do() (*AccountStatus, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the status of a Merchant Center account.",
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the statuses of the sub-accounts in your Merchant Center
// account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccountstatusesListCall) QuotaUser(quotaUser string) *AccountstatusesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccountstatusesListCall) UserIP(userIP string) *AccountstatusesListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccountstatusesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accountstatuses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.accountstatuses.list" call.
// Exactly one of *AccountstatusesListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AccountstatusesListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountstatusesListCall) Do() (*AccountstatusesListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the statuses of the sub-accounts in your Merchant Center account.",
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

// method id "content.accounttax.custombatch":

type AccounttaxCustombatchCall struct {
	s                            *Service
	accounttaxcustombatchrequest *AccounttaxCustomBatchRequest
	urlParams_                   gensupport.URLParams
	ctx_                         context.Context
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccounttaxCustombatchCall) QuotaUser(quotaUser string) *AccounttaxCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccounttaxCustombatchCall) UserIP(userIP string) *AccounttaxCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccounttaxCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accounttaxcustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounttax/batch")
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

// Do executes the "content.accounttax.custombatch" call.
// Exactly one of *AccounttaxCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *AccounttaxCustomBatchResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccounttaxCustombatchCall) Do() (*AccounttaxCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s            *Service
	merchantId   uint64
	accountId    uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the tax settings of the account.
func (r *AccounttaxService) Get(merchantId uint64, accountId uint64) *AccounttaxGetCall {
	c := &AccounttaxGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.accountId = accountId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccounttaxGetCall) QuotaUser(quotaUser string) *AccounttaxGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccounttaxGetCall) UserIP(userIP string) *AccounttaxGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccounttaxGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
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

// Do executes the "content.accounttax.get" call.
// Exactly one of *AccountTax or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountTax.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccounttaxGetCall) Do() (*AccountTax, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the tax settings of the account.",
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the tax settings of the sub-accounts in your Merchant
// Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccounttaxListCall) QuotaUser(quotaUser string) *AccounttaxListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccounttaxListCall) UserIP(userIP string) *AccounttaxListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccounttaxListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.accounttax.list" call.
// Exactly one of *AccounttaxListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AccounttaxListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccounttaxListCall) Do() (*AccounttaxListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the tax settings of the sub-accounts in your Merchant Center account.",
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

// method id "content.accounttax.patch":

type AccounttaxPatchCall struct {
	s          *Service
	merchantId uint64
	accountId  uint64
	accounttax *AccountTax
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates the tax settings of the account. This method supports
// patch semantics.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccounttaxPatchCall) QuotaUser(quotaUser string) *AccounttaxPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccounttaxPatchCall) UserIP(userIP string) *AccounttaxPatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccounttaxPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accounttax)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accounttax.patch" call.
// Exactly one of *AccountTax or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountTax.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccounttaxPatchCall) Do() (*AccountTax, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the tax settings of the account. This method supports patch semantics.",
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
	s          *Service
	merchantId uint64
	accountId  uint64
	accounttax *AccountTax
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates the tax settings of the account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AccounttaxUpdateCall) QuotaUser(quotaUser string) *AccounttaxUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AccounttaxUpdateCall) UserIP(userIP string) *AccounttaxUpdateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *AccounttaxUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accounttax)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/accounttax/{accountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"accountId":  strconv.FormatUint(c.accountId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.accounttax.update" call.
// Exactly one of *AccountTax or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AccountTax.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccounttaxUpdateCall) Do() (*AccountTax, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the tax settings of the account.",
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
	s                           *Service
	datafeedscustombatchrequest *DatafeedsCustomBatchRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedsCustombatchCall) QuotaUser(quotaUser string) *DatafeedsCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedsCustombatchCall) UserIP(userIP string) *DatafeedsCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedsCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeedscustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "datafeeds/batch")
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

// Do executes the "content.datafeeds.custombatch" call.
// Exactly one of *DatafeedsCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *DatafeedsCustomBatchResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedsCustombatchCall) Do() (*DatafeedsCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s          *Service
	merchantId uint64
	datafeedId uint64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a datafeed from your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedsDeleteCall) QuotaUser(quotaUser string) *DatafeedsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedsDeleteCall) UserIP(userIP string) *DatafeedsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.datafeeds.delete" call.
func (c *DatafeedsDeleteCall) Do() error {
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
	//   "description": "Deletes a datafeed from your Merchant Center account.",
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
	s            *Service
	merchantId   uint64
	datafeedId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves a datafeed from your Merchant Center account.
func (r *DatafeedsService) Get(merchantId uint64, datafeedId uint64) *DatafeedsGetCall {
	c := &DatafeedsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeedId = datafeedId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedsGetCall) QuotaUser(quotaUser string) *DatafeedsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedsGetCall) UserIP(userIP string) *DatafeedsGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
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

// Do executes the "content.datafeeds.get" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsGetCall) Do() (*Datafeed, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a datafeed from your Merchant Center account.",
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
	s          *Service
	merchantId uint64
	datafeed   *Datafeed
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Registers a datafeed with your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedsInsertCall) QuotaUser(quotaUser string) *DatafeedsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedsInsertCall) UserIP(userIP string) *DatafeedsInsertCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeed)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.datafeeds.insert" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsInsertCall) Do() (*Datafeed, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Registers a datafeed with your Merchant Center account.",
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the datafeeds in your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedsListCall) QuotaUser(quotaUser string) *DatafeedsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedsListCall) UserIP(userIP string) *DatafeedsListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.datafeeds.list" call.
// Exactly one of *DatafeedsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DatafeedsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedsListCall) Do() (*DatafeedsListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the datafeeds in your Merchant Center account.",
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

// method id "content.datafeeds.patch":

type DatafeedsPatchCall struct {
	s          *Service
	merchantId uint64
	datafeedId uint64
	datafeed   *Datafeed
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates a datafeed of your Merchant Center account. This
// method supports patch semantics.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedsPatchCall) QuotaUser(quotaUser string) *DatafeedsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedsPatchCall) UserIP(userIP string) *DatafeedsPatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeed)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.datafeeds.patch" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsPatchCall) Do() (*Datafeed, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a datafeed of your Merchant Center account. This method supports patch semantics.",
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
	s          *Service
	merchantId uint64
	datafeedId uint64
	datafeed   *Datafeed
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates a datafeed of your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedsUpdateCall) QuotaUser(quotaUser string) *DatafeedsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedsUpdateCall) UserIP(userIP string) *DatafeedsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeed)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeeds/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.datafeeds.update" call.
// Exactly one of *Datafeed or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Datafeed.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatafeedsUpdateCall) Do() (*Datafeed, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a datafeed of your Merchant Center account.",
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
	s                                  *Service
	datafeedstatusescustombatchrequest *DatafeedstatusesCustomBatchRequest
	urlParams_                         gensupport.URLParams
	ctx_                               context.Context
}

// Custombatch:
func (r *DatafeedstatusesService) Custombatch(datafeedstatusescustombatchrequest *DatafeedstatusesCustomBatchRequest) *DatafeedstatusesCustombatchCall {
	c := &DatafeedstatusesCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datafeedstatusescustombatchrequest = datafeedstatusescustombatchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedstatusesCustombatchCall) QuotaUser(quotaUser string) *DatafeedstatusesCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedstatusesCustombatchCall) UserIP(userIP string) *DatafeedstatusesCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedstatusesCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datafeedstatusescustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "datafeedstatuses/batch")
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

// Do executes the "content.datafeedstatuses.custombatch" call.
// Exactly one of *DatafeedstatusesCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *DatafeedstatusesCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *DatafeedstatusesCustombatchCall) Do() (*DatafeedstatusesCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	s            *Service
	merchantId   uint64
	datafeedId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the status of a datafeed from your Merchant Center
// account.
func (r *DatafeedstatusesService) Get(merchantId uint64, datafeedId uint64) *DatafeedstatusesGetCall {
	c := &DatafeedstatusesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.datafeedId = datafeedId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedstatusesGetCall) QuotaUser(quotaUser string) *DatafeedstatusesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedstatusesGetCall) UserIP(userIP string) *DatafeedstatusesGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedstatusesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeedstatuses/{datafeedId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"datafeedId": strconv.FormatUint(c.datafeedId, 10),
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

// Do executes the "content.datafeedstatuses.get" call.
// Exactly one of *DatafeedStatus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DatafeedStatus.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedstatusesGetCall) Do() (*DatafeedStatus, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the status of a datafeed from your Merchant Center account.",
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the statuses of the datafeeds in your Merchant Center
// account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatafeedstatusesListCall) QuotaUser(quotaUser string) *DatafeedstatusesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatafeedstatusesListCall) UserIP(userIP string) *DatafeedstatusesListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DatafeedstatusesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/datafeedstatuses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.datafeedstatuses.list" call.
// Exactly one of *DatafeedstatusesListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *DatafeedstatusesListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatafeedstatusesListCall) Do() (*DatafeedstatusesListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the statuses of the datafeeds in your Merchant Center account.",
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

// method id "content.inventory.custombatch":

type InventoryCustombatchCall struct {
	s                           *Service
	inventorycustombatchrequest *InventoryCustomBatchRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
}

// Custombatch: Updates price and availability for multiple products or
// stores in a single request. This operation does not update the
// expiration date of the products.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InventoryCustombatchCall) QuotaUser(quotaUser string) *InventoryCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InventoryCustombatchCall) UserIP(userIP string) *InventoryCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *InventoryCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inventorycustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "inventory/batch")
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

// Do executes the "content.inventory.custombatch" call.
// Exactly one of *InventoryCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *InventoryCustomBatchResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InventoryCustombatchCall) Do() (*InventoryCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates price and availability for multiple products or stores in a single request. This operation does not update the expiration date of the products.",
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
	s                   *Service
	merchantId          uint64
	storeCode           string
	productId           string
	inventorysetrequest *InventorySetRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// Set: Updates price and availability of a product in your Merchant
// Center account. This operation does not update the expiration date of
// the product.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InventorySetCall) QuotaUser(quotaUser string) *InventorySetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InventorySetCall) UserIP(userIP string) *InventorySetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *InventorySetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.inventorysetrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/inventory/{storeCode}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"storeCode":  c.storeCode,
		"productId":  c.productId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.inventory.set" call.
// Exactly one of *InventorySetResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *InventorySetResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InventorySetCall) Do() (*InventorySetResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates price and availability of a product in your Merchant Center account. This operation does not update the expiration date of the product.",
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
	s                        *Service
	merchantId               uint64
	orderId                  string
	ordersacknowledgerequest *OrdersAcknowledgeRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
}

// Acknowledge: Marks an order as acknowledged.
func (r *OrdersService) Acknowledge(merchantId uint64, orderId string, ordersacknowledgerequest *OrdersAcknowledgeRequest) *OrdersAcknowledgeCall {
	c := &OrdersAcknowledgeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersacknowledgerequest = ordersacknowledgerequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersAcknowledgeCall) QuotaUser(quotaUser string) *OrdersAcknowledgeCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersAcknowledgeCall) UserIP(userIP string) *OrdersAcknowledgeCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersAcknowledgeCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersacknowledgerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/acknowledge")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.acknowledge" call.
// Exactly one of *OrdersAcknowledgeResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *OrdersAcknowledgeResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersAcknowledgeCall) Do() (*OrdersAcknowledgeResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Marks an order as acknowledged.",
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
	s          *Service
	merchantId uint64
	orderId    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Advancetestorder: Sandbox only. Moves a test order from state
// "inProgress" to state "pendingShipment".
func (r *OrdersService) Advancetestorder(merchantId uint64, orderId string) *OrdersAdvancetestorderCall {
	c := &OrdersAdvancetestorderCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersAdvancetestorderCall) QuotaUser(quotaUser string) *OrdersAdvancetestorderCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersAdvancetestorderCall) UserIP(userIP string) *OrdersAdvancetestorderCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersAdvancetestorderCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/testorders/{orderId}/advance")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.advancetestorder" call.
// Exactly one of *OrdersAdvanceTestOrderResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersAdvanceTestOrderResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersAdvancetestorderCall) Do() (*OrdersAdvanceTestOrderResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sandbox only. Moves a test order from state \"inProgress\" to state \"pendingShipment\".",
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
	s                   *Service
	merchantId          uint64
	orderId             string
	orderscancelrequest *OrdersCancelRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// Cancel: Cancels all line items in an order.
func (r *OrdersService) Cancel(merchantId uint64, orderId string, orderscancelrequest *OrdersCancelRequest) *OrdersCancelCall {
	c := &OrdersCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.orderscancelrequest = orderscancelrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersCancelCall) QuotaUser(quotaUser string) *OrdersCancelCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersCancelCall) UserIP(userIP string) *OrdersCancelCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersCancelCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscancelrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.cancel" call.
// Exactly one of *OrdersCancelResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OrdersCancelResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCancelCall) Do() (*OrdersCancelResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Cancels all line items in an order.",
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
	s                           *Service
	merchantId                  uint64
	orderId                     string
	orderscancellineitemrequest *OrdersCancelLineItemRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
}

// Cancellineitem: Cancels a line item.
func (r *OrdersService) Cancellineitem(merchantId uint64, orderId string, orderscancellineitemrequest *OrdersCancelLineItemRequest) *OrdersCancellineitemCall {
	c := &OrdersCancellineitemCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.orderscancellineitemrequest = orderscancellineitemrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersCancellineitemCall) QuotaUser(quotaUser string) *OrdersCancellineitemCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersCancellineitemCall) UserIP(userIP string) *OrdersCancellineitemCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersCancellineitemCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscancellineitemrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/cancelLineItem")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.cancellineitem" call.
// Exactly one of *OrdersCancelLineItemResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersCancelLineItemResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCancellineitemCall) Do() (*OrdersCancelLineItemResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Cancels a line item.",
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
	s                            *Service
	merchantId                   uint64
	orderscreatetestorderrequest *OrdersCreateTestOrderRequest
	urlParams_                   gensupport.URLParams
	ctx_                         context.Context
}

// Createtestorder: Sandbox only. Creates a test order.
func (r *OrdersService) Createtestorder(merchantId uint64, orderscreatetestorderrequest *OrdersCreateTestOrderRequest) *OrdersCreatetestorderCall {
	c := &OrdersCreatetestorderCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderscreatetestorderrequest = orderscreatetestorderrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersCreatetestorderCall) QuotaUser(quotaUser string) *OrdersCreatetestorderCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersCreatetestorderCall) UserIP(userIP string) *OrdersCreatetestorderCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersCreatetestorderCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscreatetestorderrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/testorders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.createtestorder" call.
// Exactly one of *OrdersCreateTestOrderResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersCreateTestOrderResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCreatetestorderCall) Do() (*OrdersCreateTestOrderResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sandbox only. Creates a test order.",
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
	s                        *Service
	orderscustombatchrequest *OrdersCustomBatchRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
}

// Custombatch: Retrieves or modifies multiple orders in a single
// request.
func (r *OrdersService) Custombatch(orderscustombatchrequest *OrdersCustomBatchRequest) *OrdersCustombatchCall {
	c := &OrdersCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.orderscustombatchrequest = orderscustombatchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersCustombatchCall) QuotaUser(quotaUser string) *OrdersCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersCustombatchCall) UserIP(userIP string) *OrdersCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.orderscustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "orders/batch")
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

// Do executes the "content.orders.custombatch" call.
// Exactly one of *OrdersCustomBatchResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *OrdersCustomBatchResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersCustombatchCall) Do() (*OrdersCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves or modifies multiple orders in a single request.",
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
	s            *Service
	merchantId   uint64
	orderId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves an order from your Merchant Center account.
func (r *OrdersService) Get(merchantId uint64, orderId string) *OrdersGetCall {
	c := &OrdersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersGetCall) QuotaUser(quotaUser string) *OrdersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersGetCall) UserIP(userIP string) *OrdersGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
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

// Do executes the "content.orders.get" call.
// Exactly one of *Order or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Order.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OrdersGetCall) Do() (*Order, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves an order from your Merchant Center account.",
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
	s               *Service
	merchantId      uint64
	merchantOrderId string
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
}

// Getbymerchantorderid: Retrieves an order using merchant order id.
func (r *OrdersService) Getbymerchantorderid(merchantId uint64, merchantOrderId string) *OrdersGetbymerchantorderidCall {
	c := &OrdersGetbymerchantorderidCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.merchantOrderId = merchantOrderId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersGetbymerchantorderidCall) QuotaUser(quotaUser string) *OrdersGetbymerchantorderidCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersGetbymerchantorderidCall) UserIP(userIP string) *OrdersGetbymerchantorderidCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersGetbymerchantorderidCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/ordersbymerchantid/{merchantOrderId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId":      strconv.FormatUint(c.merchantId, 10),
		"merchantOrderId": c.merchantOrderId,
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

// Do executes the "content.orders.getbymerchantorderid" call.
// Exactly one of *OrdersGetByMerchantOrderIdResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersGetByMerchantOrderIdResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrdersGetbymerchantorderidCall) Do() (*OrdersGetByMerchantOrderIdResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves an order using merchant order id.",
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
	s            *Service
	merchantId   uint64
	templateName string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Gettestordertemplate: Sandbox only. Retrieves an order template that
// can be used to quickly create a new order in sandbox.
func (r *OrdersService) Gettestordertemplate(merchantId uint64, templateName string) *OrdersGettestordertemplateCall {
	c := &OrdersGettestordertemplateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.templateName = templateName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersGettestordertemplateCall) QuotaUser(quotaUser string) *OrdersGettestordertemplateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersGettestordertemplateCall) UserIP(userIP string) *OrdersGettestordertemplateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersGettestordertemplateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/testordertemplates/{templateName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId":   strconv.FormatUint(c.merchantId, 10),
		"templateName": c.templateName,
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

// Do executes the "content.orders.gettestordertemplate" call.
// Exactly one of *OrdersGetTestOrderTemplateResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersGetTestOrderTemplateResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrdersGettestordertemplateCall) Do() (*OrdersGetTestOrderTemplateResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sandbox only. Retrieves an order template that can be used to quickly create a new order in sandbox.",
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
	//         "template2"
	//       ],
	//       "enumDescriptions": [
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the orders in your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersListCall) QuotaUser(quotaUser string) *OrdersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersListCall) UserIP(userIP string) *OrdersListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.orders.list" call.
// Exactly one of *OrdersListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OrdersListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersListCall) Do() (*OrdersListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the orders in your Merchant Center account.",
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

// method id "content.orders.refund":

type OrdersRefundCall struct {
	s                   *Service
	merchantId          uint64
	orderId             string
	ordersrefundrequest *OrdersRefundRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// Refund: Refund a portion of the order, up to the full amount paid.
func (r *OrdersService) Refund(merchantId uint64, orderId string, ordersrefundrequest *OrdersRefundRequest) *OrdersRefundCall {
	c := &OrdersRefundCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersrefundrequest = ordersrefundrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersRefundCall) QuotaUser(quotaUser string) *OrdersRefundCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersRefundCall) UserIP(userIP string) *OrdersRefundCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersRefundCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersrefundrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/refund")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.refund" call.
// Exactly one of *OrdersRefundResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OrdersRefundResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersRefundCall) Do() (*OrdersRefundResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Refund a portion of the order, up to the full amount paid.",
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
	s                           *Service
	merchantId                  uint64
	orderId                     string
	ordersreturnlineitemrequest *OrdersReturnLineItemRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
}

// Returnlineitem: Returns a line item.
func (r *OrdersService) Returnlineitem(merchantId uint64, orderId string, ordersreturnlineitemrequest *OrdersReturnLineItemRequest) *OrdersReturnlineitemCall {
	c := &OrdersReturnlineitemCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersreturnlineitemrequest = ordersreturnlineitemrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersReturnlineitemCall) QuotaUser(quotaUser string) *OrdersReturnlineitemCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersReturnlineitemCall) UserIP(userIP string) *OrdersReturnlineitemCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersReturnlineitemCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersreturnlineitemrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/returnLineItem")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.returnlineitem" call.
// Exactly one of *OrdersReturnLineItemResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersReturnLineItemResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersReturnlineitemCall) Do() (*OrdersReturnLineItemResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a line item.",
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
	s                          *Service
	merchantId                 uint64
	orderId                    string
	ordersshiplineitemsrequest *OrdersShipLineItemsRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
}

// Shiplineitems: Marks line item(s) as shipped.
func (r *OrdersService) Shiplineitems(merchantId uint64, orderId string, ordersshiplineitemsrequest *OrdersShipLineItemsRequest) *OrdersShiplineitemsCall {
	c := &OrdersShiplineitemsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersshiplineitemsrequest = ordersshiplineitemsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersShiplineitemsCall) QuotaUser(quotaUser string) *OrdersShiplineitemsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersShiplineitemsCall) UserIP(userIP string) *OrdersShiplineitemsCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersShiplineitemsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersshiplineitemsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/shipLineItems")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.shiplineitems" call.
// Exactly one of *OrdersShipLineItemsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *OrdersShipLineItemsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersShiplineitemsCall) Do() (*OrdersShipLineItemsResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Marks line item(s) as shipped.",
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
	s                                  *Service
	merchantId                         uint64
	orderId                            string
	ordersupdatemerchantorderidrequest *OrdersUpdateMerchantOrderIdRequest
	urlParams_                         gensupport.URLParams
	ctx_                               context.Context
}

// Updatemerchantorderid: Updates the merchant order ID for a given
// order.
func (r *OrdersService) Updatemerchantorderid(merchantId uint64, orderId string, ordersupdatemerchantorderidrequest *OrdersUpdateMerchantOrderIdRequest) *OrdersUpdatemerchantorderidCall {
	c := &OrdersUpdatemerchantorderidCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersupdatemerchantorderidrequest = ordersupdatemerchantorderidrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersUpdatemerchantorderidCall) QuotaUser(quotaUser string) *OrdersUpdatemerchantorderidCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersUpdatemerchantorderidCall) UserIP(userIP string) *OrdersUpdatemerchantorderidCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersUpdatemerchantorderidCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersupdatemerchantorderidrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/updateMerchantOrderId")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.updatemerchantorderid" call.
// Exactly one of *OrdersUpdateMerchantOrderIdResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersUpdateMerchantOrderIdResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrdersUpdatemerchantorderidCall) Do() (*OrdersUpdateMerchantOrderIdResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the merchant order ID for a given order.",
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
	s                           *Service
	merchantId                  uint64
	orderId                     string
	ordersupdateshipmentrequest *OrdersUpdateShipmentRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
}

// Updateshipment: Updates a shipment's status, carrier, and/or tracking
// ID.
func (r *OrdersService) Updateshipment(merchantId uint64, orderId string, ordersupdateshipmentrequest *OrdersUpdateShipmentRequest) *OrdersUpdateshipmentCall {
	c := &OrdersUpdateshipmentCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.orderId = orderId
	c.ordersupdateshipmentrequest = ordersupdateshipmentrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OrdersUpdateshipmentCall) QuotaUser(quotaUser string) *OrdersUpdateshipmentCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OrdersUpdateshipmentCall) UserIP(userIP string) *OrdersUpdateshipmentCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *OrdersUpdateshipmentCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ordersupdateshipmentrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/orders/{orderId}/updateShipment")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"orderId":    c.orderId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.orders.updateshipment" call.
// Exactly one of *OrdersUpdateShipmentResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *OrdersUpdateShipmentResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrdersUpdateshipmentCall) Do() (*OrdersUpdateShipmentResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a shipment's status, carrier, and/or tracking ID.",
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
	s                          *Service
	productscustombatchrequest *ProductsCustomBatchRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
}

// Custombatch: Retrieves, inserts, and deletes multiple products in a
// single request.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsCustombatchCall) QuotaUser(quotaUser string) *ProductsCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsCustombatchCall) UserIP(userIP string) *ProductsCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productscustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "products/batch")
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

// Do executes the "content.products.custombatch" call.
// Exactly one of *ProductsCustomBatchResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ProductsCustomBatchResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsCustombatchCall) Do() (*ProductsCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves, inserts, and deletes multiple products in a single request.",
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
	s          *Service
	merchantId uint64
	productId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a product from your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsDeleteCall) QuotaUser(quotaUser string) *ProductsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsDeleteCall) UserIP(userIP string) *ProductsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"productId":  c.productId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.products.delete" call.
func (c *ProductsDeleteCall) Do() error {
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
	//   "description": "Deletes a product from your Merchant Center account.",
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
	s            *Service
	merchantId   uint64
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves a product from your Merchant Center account.
func (r *ProductsService) Get(merchantId uint64, productId string) *ProductsGetCall {
	c := &ProductsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.productId = productId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsGetCall) QuotaUser(quotaUser string) *ProductsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsGetCall) UserIP(userIP string) *ProductsGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"productId":  c.productId,
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

// Do executes the "content.products.get" call.
// Exactly one of *Product or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Product.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProductsGetCall) Do() (*Product, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a product from your Merchant Center account.",
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
	s          *Service
	merchantId uint64
	product    *Product
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Uploads a product to your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsInsertCall) QuotaUser(quotaUser string) *ProductsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsInsertCall) UserIP(userIP string) *ProductsInsertCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.product)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "content.products.insert" call.
// Exactly one of *Product or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Product.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProductsInsertCall) Do() (*Product, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Uploads a product to your Merchant Center account.",
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the products in your Merchant Center account.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsListCall) QuotaUser(quotaUser string) *ProductsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsListCall) UserIP(userIP string) *ProductsListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/products")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.products.list" call.
// Exactly one of *ProductsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ProductsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsListCall) Do() (*ProductsListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the products in your Merchant Center account.",
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

// method id "content.productstatuses.custombatch":

type ProductstatusesCustombatchCall struct {
	s                                 *Service
	productstatusescustombatchrequest *ProductstatusesCustomBatchRequest
	urlParams_                        gensupport.URLParams
	ctx_                              context.Context
}

// Custombatch: Gets the statuses of multiple products in a single
// request.
func (r *ProductstatusesService) Custombatch(productstatusescustombatchrequest *ProductstatusesCustomBatchRequest) *ProductstatusesCustombatchCall {
	c := &ProductstatusesCustombatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productstatusescustombatchrequest = productstatusescustombatchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductstatusesCustombatchCall) QuotaUser(quotaUser string) *ProductstatusesCustombatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductstatusesCustombatchCall) UserIP(userIP string) *ProductstatusesCustombatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductstatusesCustombatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productstatusescustombatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "productstatuses/batch")
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

// Do executes the "content.productstatuses.custombatch" call.
// Exactly one of *ProductstatusesCustomBatchResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ProductstatusesCustomBatchResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProductstatusesCustombatchCall) Do() (*ProductstatusesCustomBatchResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the statuses of multiple products in a single request.",
	//   "httpMethod": "POST",
	//   "id": "content.productstatuses.custombatch",
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
	s            *Service
	merchantId   uint64
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the status of a product from your Merchant Center account.
func (r *ProductstatusesService) Get(merchantId uint64, productId string) *ProductstatusesGetCall {
	c := &ProductstatusesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
	c.productId = productId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductstatusesGetCall) QuotaUser(quotaUser string) *ProductstatusesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductstatusesGetCall) UserIP(userIP string) *ProductstatusesGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductstatusesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/productstatuses/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
		"productId":  c.productId,
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

// Do executes the "content.productstatuses.get" call.
// Exactly one of *ProductStatus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProductStatus.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductstatusesGetCall) Do() (*ProductStatus, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the status of a product from your Merchant Center account.",
	//   "httpMethod": "GET",
	//   "id": "content.productstatuses.get",
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
	s            *Service
	merchantId   uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the statuses of the products in your Merchant Center
// account.
func (r *ProductstatusesService) List(merchantId uint64) *ProductstatusesListCall {
	c := &ProductstatusesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.merchantId = merchantId
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductstatusesListCall) QuotaUser(quotaUser string) *ProductstatusesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductstatusesListCall) UserIP(userIP string) *ProductstatusesListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductstatusesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{merchantId}/productstatuses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"merchantId": strconv.FormatUint(c.merchantId, 10),
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

// Do executes the "content.productstatuses.list" call.
// Exactly one of *ProductstatusesListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ProductstatusesListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductstatusesListCall) Do() (*ProductstatusesListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the statuses of the products in your Merchant Center account.",
	//   "httpMethod": "GET",
	//   "id": "content.productstatuses.list",
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
