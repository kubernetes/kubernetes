// Package adexchangebuyer2 provides access to the Ad Exchange Buyer API II.
//
// See https://developers.google.com/ad-exchange/buyer-rest/reference/rest/
//
// Usage example:
//
//   import "google.golang.org/api/adexchangebuyer2/v2beta1"
//   ...
//   adexchangebuyer2Service, err := adexchangebuyer2.New(oauthHttpClient)
package adexchangebuyer2 // import "google.golang.org/api/adexchangebuyer2/v2beta1"

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

const apiId = "adexchangebuyer2:v2beta1"
const apiName = "adexchangebuyer2"
const apiVersion = "v2beta1"
const basePath = "https://adexchangebuyer.googleapis.com/"

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
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Accounts *AccountsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAccountsService(s *Service) *AccountsService {
	rs := &AccountsService{s: s}
	rs.Clients = NewAccountsClientsService(s)
	rs.Creatives = NewAccountsCreativesService(s)
	rs.FilterSets = NewAccountsFilterSetsService(s)
	return rs
}

type AccountsService struct {
	s *Service

	Clients *AccountsClientsService

	Creatives *AccountsCreativesService

	FilterSets *AccountsFilterSetsService
}

func NewAccountsClientsService(s *Service) *AccountsClientsService {
	rs := &AccountsClientsService{s: s}
	rs.Invitations = NewAccountsClientsInvitationsService(s)
	rs.Users = NewAccountsClientsUsersService(s)
	return rs
}

type AccountsClientsService struct {
	s *Service

	Invitations *AccountsClientsInvitationsService

	Users *AccountsClientsUsersService
}

func NewAccountsClientsInvitationsService(s *Service) *AccountsClientsInvitationsService {
	rs := &AccountsClientsInvitationsService{s: s}
	return rs
}

type AccountsClientsInvitationsService struct {
	s *Service
}

func NewAccountsClientsUsersService(s *Service) *AccountsClientsUsersService {
	rs := &AccountsClientsUsersService{s: s}
	return rs
}

type AccountsClientsUsersService struct {
	s *Service
}

func NewAccountsCreativesService(s *Service) *AccountsCreativesService {
	rs := &AccountsCreativesService{s: s}
	rs.DealAssociations = NewAccountsCreativesDealAssociationsService(s)
	return rs
}

type AccountsCreativesService struct {
	s *Service

	DealAssociations *AccountsCreativesDealAssociationsService
}

func NewAccountsCreativesDealAssociationsService(s *Service) *AccountsCreativesDealAssociationsService {
	rs := &AccountsCreativesDealAssociationsService{s: s}
	return rs
}

type AccountsCreativesDealAssociationsService struct {
	s *Service
}

func NewAccountsFilterSetsService(s *Service) *AccountsFilterSetsService {
	rs := &AccountsFilterSetsService{s: s}
	rs.BidMetrics = NewAccountsFilterSetsBidMetricsService(s)
	rs.BidResponseErrors = NewAccountsFilterSetsBidResponseErrorsService(s)
	rs.BidResponsesWithoutBids = NewAccountsFilterSetsBidResponsesWithoutBidsService(s)
	rs.FilteredBidRequests = NewAccountsFilterSetsFilteredBidRequestsService(s)
	rs.FilteredBids = NewAccountsFilterSetsFilteredBidsService(s)
	rs.ImpressionMetrics = NewAccountsFilterSetsImpressionMetricsService(s)
	rs.LosingBids = NewAccountsFilterSetsLosingBidsService(s)
	rs.NonBillableWinningBids = NewAccountsFilterSetsNonBillableWinningBidsService(s)
	return rs
}

type AccountsFilterSetsService struct {
	s *Service

	BidMetrics *AccountsFilterSetsBidMetricsService

	BidResponseErrors *AccountsFilterSetsBidResponseErrorsService

	BidResponsesWithoutBids *AccountsFilterSetsBidResponsesWithoutBidsService

	FilteredBidRequests *AccountsFilterSetsFilteredBidRequestsService

	FilteredBids *AccountsFilterSetsFilteredBidsService

	ImpressionMetrics *AccountsFilterSetsImpressionMetricsService

	LosingBids *AccountsFilterSetsLosingBidsService

	NonBillableWinningBids *AccountsFilterSetsNonBillableWinningBidsService
}

func NewAccountsFilterSetsBidMetricsService(s *Service) *AccountsFilterSetsBidMetricsService {
	rs := &AccountsFilterSetsBidMetricsService{s: s}
	return rs
}

type AccountsFilterSetsBidMetricsService struct {
	s *Service
}

func NewAccountsFilterSetsBidResponseErrorsService(s *Service) *AccountsFilterSetsBidResponseErrorsService {
	rs := &AccountsFilterSetsBidResponseErrorsService{s: s}
	return rs
}

type AccountsFilterSetsBidResponseErrorsService struct {
	s *Service
}

func NewAccountsFilterSetsBidResponsesWithoutBidsService(s *Service) *AccountsFilterSetsBidResponsesWithoutBidsService {
	rs := &AccountsFilterSetsBidResponsesWithoutBidsService{s: s}
	return rs
}

type AccountsFilterSetsBidResponsesWithoutBidsService struct {
	s *Service
}

func NewAccountsFilterSetsFilteredBidRequestsService(s *Service) *AccountsFilterSetsFilteredBidRequestsService {
	rs := &AccountsFilterSetsFilteredBidRequestsService{s: s}
	return rs
}

type AccountsFilterSetsFilteredBidRequestsService struct {
	s *Service
}

func NewAccountsFilterSetsFilteredBidsService(s *Service) *AccountsFilterSetsFilteredBidsService {
	rs := &AccountsFilterSetsFilteredBidsService{s: s}
	rs.Creatives = NewAccountsFilterSetsFilteredBidsCreativesService(s)
	rs.Details = NewAccountsFilterSetsFilteredBidsDetailsService(s)
	return rs
}

type AccountsFilterSetsFilteredBidsService struct {
	s *Service

	Creatives *AccountsFilterSetsFilteredBidsCreativesService

	Details *AccountsFilterSetsFilteredBidsDetailsService
}

func NewAccountsFilterSetsFilteredBidsCreativesService(s *Service) *AccountsFilterSetsFilteredBidsCreativesService {
	rs := &AccountsFilterSetsFilteredBidsCreativesService{s: s}
	return rs
}

type AccountsFilterSetsFilteredBidsCreativesService struct {
	s *Service
}

func NewAccountsFilterSetsFilteredBidsDetailsService(s *Service) *AccountsFilterSetsFilteredBidsDetailsService {
	rs := &AccountsFilterSetsFilteredBidsDetailsService{s: s}
	return rs
}

type AccountsFilterSetsFilteredBidsDetailsService struct {
	s *Service
}

func NewAccountsFilterSetsImpressionMetricsService(s *Service) *AccountsFilterSetsImpressionMetricsService {
	rs := &AccountsFilterSetsImpressionMetricsService{s: s}
	return rs
}

type AccountsFilterSetsImpressionMetricsService struct {
	s *Service
}

func NewAccountsFilterSetsLosingBidsService(s *Service) *AccountsFilterSetsLosingBidsService {
	rs := &AccountsFilterSetsLosingBidsService{s: s}
	return rs
}

type AccountsFilterSetsLosingBidsService struct {
	s *Service
}

func NewAccountsFilterSetsNonBillableWinningBidsService(s *Service) *AccountsFilterSetsNonBillableWinningBidsService {
	rs := &AccountsFilterSetsNonBillableWinningBidsService{s: s}
	return rs
}

type AccountsFilterSetsNonBillableWinningBidsService struct {
	s *Service
}

// AbsoluteDateRange: An absolute date range, specified by its start
// date and end date.
// The supported range of dates begins 30 days before today and ends
// today.
// Validity checked upon filter set creation. If a filter set with an
// absolute
// date range is run at a later date more than 30 days after start_date,
// it will
// fail.
type AbsoluteDateRange struct {
	// EndDate: The end date of the range (inclusive).
	// Must be within the 30 days leading up to current date, and must be
	// equal to
	// or after start_date.
	EndDate *Date `json:"endDate,omitempty"`

	// StartDate: The start date of the range (inclusive).
	// Must be within the 30 days leading up to current date, and must be
	// equal to
	// or before end_date.
	StartDate *Date `json:"startDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndDate") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AbsoluteDateRange) MarshalJSON() ([]byte, error) {
	type noMethod AbsoluteDateRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddDealAssociationRequest: A request for associating a deal and a
// creative.
type AddDealAssociationRequest struct {
	// Association: The association between a creative and a deal that
	// should be added.
	Association *CreativeDealAssociation `json:"association,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Association") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Association") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddDealAssociationRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddDealAssociationRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppContext: @OutputOnly The app type the restriction applies to for
// mobile device.
type AppContext struct {
	// AppTypes: The app types this restriction applies to.
	//
	// Possible values:
	//   "NATIVE" - Native app context.
	//   "WEB" - Mobile web app context.
	AppTypes []string `json:"appTypes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AppTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppTypes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppContext) MarshalJSON() ([]byte, error) {
	type noMethod AppContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuctionContext: @OutputOnly The auction type the restriction applies
// to.
type AuctionContext struct {
	// AuctionTypes: The auction types this restriction applies to.
	//
	// Possible values:
	//   "OPEN_AUCTION" - The restriction applies to open auction.
	//   "DIRECT_DEALS" - The restriction applies to direct deals.
	AuctionTypes []string `json:"auctionTypes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuctionTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuctionTypes") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuctionContext) MarshalJSON() ([]byte, error) {
	type noMethod AuctionContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BidMetricsRow: The set of metrics that are measured in numbers of
// bids, representing how
// many bids with the specified dimension values were considered
// eligible at
// each stage of the bidding funnel;
type BidMetricsRow struct {
	// Bids: The number of bids that Ad Exchange received from the buyer.
	Bids *MetricValue `json:"bids,omitempty"`

	// BidsInAuction: The number of bids that were permitted to compete in
	// the auction.
	BidsInAuction *MetricValue `json:"bidsInAuction,omitempty"`

	// BilledImpressions: The number of bids for which the buyer was billed.
	BilledImpressions *MetricValue `json:"billedImpressions,omitempty"`

	// ImpressionsWon: The number of bids that won an impression.
	ImpressionsWon *MetricValue `json:"impressionsWon,omitempty"`

	// MeasurableImpressions: The number of bids for which the corresponding
	// impression was measurable
	// for viewability (as defined by Active View).
	MeasurableImpressions *MetricValue `json:"measurableImpressions,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// ViewableImpressions: The number of bids for which the corresponding
	// impression was viewable (as
	// defined by Active View).
	ViewableImpressions *MetricValue `json:"viewableImpressions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Bids") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bids") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BidMetricsRow) MarshalJSON() ([]byte, error) {
	type noMethod BidMetricsRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BidResponseWithoutBidsStatusRow: The number of impressions with the
// specified dimension values that were
// considered to have no applicable bids, as described by the specified
// status.
type BidResponseWithoutBidsStatusRow struct {
	// ImpressionCount: The number of impressions for which there was a bid
	// response with the
	// specified status.
	ImpressionCount *MetricValue `json:"impressionCount,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// Status: The status specifying why the bid responses were considered
	// to have no
	// applicable bids.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - A placeholder for an undefined status.
	// This value will never be returned in responses.
	//   "RESPONSES_WITHOUT_BIDS" - The response had no bids.
	//   "RESPONSES_WITHOUT_BIDS_FOR_ACCOUNT" - The response had no bids for
	// the specified account, though it may have
	// included bids on behalf of other accounts.
	//   "RESPONSES_WITHOUT_BIDS_FOR_DEAL" - The response had no bids for
	// the specified deal, though it may have
	// included bids on other deals on behalf of the account to which the
	// deal
	// belongs.
	Status string `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ImpressionCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ImpressionCount") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BidResponseWithoutBidsStatusRow) MarshalJSON() ([]byte, error) {
	type noMethod BidResponseWithoutBidsStatusRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CalloutStatusRow: The number of impressions with the specified
// dimension values where the
// corresponding bid request or bid response was not successful, as
// described by
// the specified callout status.
type CalloutStatusRow struct {
	// CalloutStatusId: The ID of the callout status.
	// See
	// [callout-status-codes](https://developers.google.com/ad-exchange/rtb/d
	// ownloads/callout-status-codes).
	CalloutStatusId int64 `json:"calloutStatusId,omitempty"`

	// ImpressionCount: The number of impressions for which there was a bid
	// request or bid response
	// with the specified callout status.
	ImpressionCount *MetricValue `json:"impressionCount,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CalloutStatusId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CalloutStatusId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CalloutStatusRow) MarshalJSON() ([]byte, error) {
	type noMethod CalloutStatusRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Client: A client resource represents a client buyer&mdash;an
// agency,
// a brand, or an advertiser customer of the sponsor buyer.
// Users associated with the client buyer have restricted access to
// the Ad Exchange Marketplace and certain other sections
// of the Ad Exchange Buyer UI based on the role
// granted to the client buyer.
// All fields are required unless otherwise specified.
type Client struct {
	// ClientAccountId: The globally-unique numerical ID of the client.
	// The value of this field is ignored in create and update operations.
	ClientAccountId int64 `json:"clientAccountId,omitempty,string"`

	// ClientName: Name used to represent this client to publishers.
	// You may have multiple clients that map to the same entity,
	// but for each client the combination of `clientName` and entity
	// must be unique.
	// You can specify this field as empty.
	ClientName string `json:"clientName,omitempty"`

	// EntityId: Numerical identifier of the client entity.
	// The entity can be an advertiser, a brand, or an agency.
	// This identifier is unique among all the entities with the same
	// type.
	//
	// A list of all known advertisers with their identifiers is available
	// in
	// the
	// [advertisers.txt](https://storage.googleapis.com/adx-rtb-dictionar
	// ies/advertisers.txt)
	// file.
	//
	// A list of all known brands with their identifiers is available in
	// the
	// [brands.txt](https://storage.googleapis.com/adx-rtb-dictionaries/b
	// rands.txt)
	// file.
	//
	// A list of all known agencies with their identifiers is available in
	// the
	// [agencies.txt](https://storage.googleapis.com/adx-rtb-dictionaries
	// /agencies.txt)
	// file.
	EntityId int64 `json:"entityId,omitempty,string"`

	// EntityName: The name of the entity. This field is automatically
	// fetched based on
	// the type and ID.
	// The value of this field is ignored in create and update operations.
	EntityName string `json:"entityName,omitempty"`

	// EntityType: The type of the client entity: `ADVERTISER`, `BRAND`, or
	// `AGENCY`.
	//
	// Possible values:
	//   "ENTITY_TYPE_UNSPECIFIED" - A placeholder for an undefined client
	// entity type. Should not be used.
	//   "ADVERTISER" - An advertiser.
	//   "BRAND" - A brand.
	//   "AGENCY" - An advertising agency.
	EntityType string `json:"entityType,omitempty"`

	// Role: The role which is assigned to the client buyer. Each role
	// implies a set of
	// permissions granted to the client. Must be one of
	// `CLIENT_DEAL_VIEWER`,
	// `CLIENT_DEAL_NEGOTIATOR` or `CLIENT_DEAL_APPROVER`.
	//
	// Possible values:
	//   "CLIENT_ROLE_UNSPECIFIED" - A placeholder for an undefined client
	// role.
	//   "CLIENT_DEAL_VIEWER" - Users associated with this client can see
	// publisher deal offers
	// in the Marketplace.
	// They can neither negotiate proposals nor approve deals.
	// If this client is visible to publishers, they can send deal
	// proposals
	// to this client.
	//   "CLIENT_DEAL_NEGOTIATOR" - Users associated with this client can
	// respond to deal proposals
	// sent to them by publishers. They can also initiate deal proposals
	// of their own.
	//   "CLIENT_DEAL_APPROVER" - Users associated with this client can
	// approve eligible deals
	// on your behalf. Some deals may still explicitly require
	// publisher
	// finalization. If this role is not selected, the sponsor buyer
	// will need to manually approve each of their deals.
	Role string `json:"role,omitempty"`

	// Status: The status of the client buyer.
	//
	// Possible values:
	//   "CLIENT_STATUS_UNSPECIFIED" - A placeholder for an undefined client
	// status.
	//   "DISABLED" - A client that is currently disabled.
	//   "ACTIVE" - A client that is currently active.
	Status string `json:"status,omitempty"`

	// VisibleToSeller: Whether the client buyer will be visible to sellers.
	VisibleToSeller bool `json:"visibleToSeller,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClientAccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientAccountId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Client) MarshalJSON() ([]byte, error) {
	type noMethod Client
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClientUser: A client user is created under a client buyer and has
// restricted access to
// the Ad Exchange Marketplace and certain other sections
// of the Ad Exchange Buyer UI based on the role
// granted to the associated client buyer.
//
// The only way a new client user can be created is via accepting
// an
// email invitation
// (see the
// accounts.clients.invitations.create
// method).
//
// All fields are required unless otherwise specified.
type ClientUser struct {
	// ClientAccountId: Numerical account ID of the client buyer
	// with which the user is associated; the
	// buyer must be a client of the current sponsor buyer.
	// The value of this field is ignored in an update operation.
	ClientAccountId int64 `json:"clientAccountId,omitempty,string"`

	// Email: User's email address. The value of this field
	// is ignored in an update operation.
	Email string `json:"email,omitempty"`

	// Status: The status of the client user.
	//
	// Possible values:
	//   "USER_STATUS_UNSPECIFIED" - A placeholder for an undefined user
	// status.
	//   "PENDING" - A user who was already created but hasn't accepted the
	// invitation yet.
	//   "ACTIVE" - A user that is currently active.
	//   "DISABLED" - A user that is currently disabled.
	Status string `json:"status,omitempty"`

	// UserId: The unique numerical ID of the client user
	// that has accepted an invitation.
	// The value of this field is ignored in an update operation.
	UserId int64 `json:"userId,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClientAccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientAccountId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ClientUser) MarshalJSON() ([]byte, error) {
	type noMethod ClientUser
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClientUserInvitation: An invitation for a new client user to get
// access to the Ad Exchange
// Buyer UI.
// All fields are required unless otherwise specified.
type ClientUserInvitation struct {
	// ClientAccountId: Numerical account ID of the client buyer
	// that the invited user is associated with.
	// The value of this field is ignored in create operations.
	ClientAccountId int64 `json:"clientAccountId,omitempty,string"`

	// Email: The email address to which the invitation is sent.
	// Email
	// addresses should be unique among all client users under each
	// sponsor
	// buyer.
	Email string `json:"email,omitempty"`

	// InvitationId: The unique numerical ID of the invitation that is sent
	// to the user.
	// The value of this field is ignored in create operations.
	InvitationId int64 `json:"invitationId,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClientAccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientAccountId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ClientUserInvitation) MarshalJSON() ([]byte, error) {
	type noMethod ClientUserInvitation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Correction: @OutputOnly Shows any corrections that were applied to
// this creative.
type Correction struct {
	// Contexts: The contexts for the correction.
	Contexts []*ServingContext `json:"contexts,omitempty"`

	// Details: Additional details about what was corrected.
	Details []string `json:"details,omitempty"`

	// Type: The type of correction that was applied to the creative.
	//
	// Possible values:
	//   "CORRECTION_TYPE_UNSPECIFIED" - The correction type is unknown.
	// Refer to the details for more information.
	//   "VENDOR_IDS_ADDED" - The ad's declared vendors did not match the
	// vendors that were detected.
	// The detected vendors were added.
	//   "SSL_ATTRIBUTE_REMOVED" - The ad had the SSL attribute declared but
	// was not SSL-compliant.
	// The SSL attribute was removed.
	//   "FLASH_FREE_ATTRIBUTE_REMOVED" - The ad was declared as Flash-free
	// but contained Flash, so the Flash-free
	// attribute was removed.
	//   "FLASH_FREE_ATTRIBUTE_ADDED" - The ad was not declared as
	// Flash-free but it did not reference any flash
	// content, so the Flash-free attribute was added.
	//   "REQUIRED_ATTRIBUTE_ADDED" - The ad did not declare a required
	// creative attribute.
	// The attribute was added.
	//   "REQUIRED_VENDOR_ADDED" - The ad did not declare a required
	// technology vendor.
	// The technology vendor was added.
	//   "SSL_ATTRIBUTE_ADDED" - The ad did not declare the SSL attribute
	// but was SSL-compliant, so the
	// SSL attribute was added.
	//   "IN_BANNER_VIDEO_ATTRIBUTE_ADDED" - Properties consistent with
	// In-banner video were found, so an
	// In-Banner Video attribute was added.
	//   "MRAID_ATTRIBUTE_ADDED" - The ad makes calls to the MRAID API so
	// the MRAID attribute was added.
	//   "FLASH_ATTRIBUTE_REMOVED" - The ad unnecessarily declared the Flash
	// attribute, so the Flash attribute
	// was removed.
	//   "VIDEO_IN_SNIPPET_ATTRIBUTE_ADDED" - The ad contains video content.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Contexts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Contexts") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Correction) MarshalJSON() ([]byte, error) {
	type noMethod Correction
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Creative: A creative and its classification data.
type Creative struct {
	// AccountId: The account that this creative belongs to.
	// Can be used to filter the response of the
	// creatives.list
	// method.
	AccountId string `json:"accountId,omitempty"`

	// AdChoicesDestinationUrl: The link to AdChoices destination page.
	AdChoicesDestinationUrl string `json:"adChoicesDestinationUrl,omitempty"`

	// AdvertiserName: The name of the company being advertised in the
	// creative.
	AdvertiserName string `json:"advertiserName,omitempty"`

	// AgencyId: The agency ID for this creative.
	AgencyId int64 `json:"agencyId,omitempty,string"`

	// ApiUpdateTime: @OutputOnly The last update timestamp of the creative
	// via API.
	ApiUpdateTime string `json:"apiUpdateTime,omitempty"`

	// Attributes: All attributes for the ads that may be shown from this
	// creative.
	// Can be used to filter the response of the
	// creatives.list
	// method.
	//
	// Possible values:
	//   "ATTRIBUTE_UNSPECIFIED" - Do not use. This is a placeholder value
	// only.
	//   "IS_TAGGED" - The creative is tagged.
	//   "IS_COOKIE_TARGETED" - The creative is cookie targeted.
	//   "IS_USER_INTEREST_TARGETED" - The creative is user interest
	// targeted.
	//   "EXPANDING_DIRECTION_NONE" - The creative does not expand.
	//   "EXPANDING_DIRECTION_UP" - The creative expands up.
	//   "EXPANDING_DIRECTION_DOWN" - The creative expands down.
	//   "EXPANDING_DIRECTION_LEFT" - The creative expands left.
	//   "EXPANDING_DIRECTION_RIGHT" - The creative expands right.
	//   "EXPANDING_DIRECTION_UP_LEFT" - The creative expands up and left.
	//   "EXPANDING_DIRECTION_UP_RIGHT" - The creative expands up and right.
	//   "EXPANDING_DIRECTION_DOWN_LEFT" - The creative expands down and
	// left.
	//   "EXPANDING_DIRECTION_DOWN_RIGHT" - The creative expands down and
	// right.
	//   "EXPANDING_DIRECTION_UP_OR_DOWN" - The creative expands up or down.
	//   "EXPANDING_DIRECTION_LEFT_OR_RIGHT" - The creative expands left or
	// right.
	//   "EXPANDING_DIRECTION_ANY_DIAGONAL" - The creative expands on any
	// diagonal.
	//   "EXPANDING_ACTION_ROLLOVER_TO_EXPAND" - The creative expands when
	// rolled over.
	//   "INSTREAM_VAST_VIDEO_TYPE_VPAID_FLASH" - The instream vast video
	// type is vpaid flash.
	//   "RICH_MEDIA_CAPABILITY_TYPE_MRAID" - The creative is MRAID
	//   "RICH_MEDIA_CAPABILITY_TYPE_SSL" - The creative is SSL.
	//   "RICH_MEDIA_CAPABILITY_TYPE_INTERSTITIAL" - The creative is an
	// interstitial.
	//   "NATIVE_ELIGIBILITY_ELIGIBLE" - The creative is eligible for
	// native.
	//   "NATIVE_ELIGIBILITY_NOT_ELIGIBLE" - The creative is not eligible
	// for native.
	//   "RENDERING_SIZELESS_ADX" - The creative can dynamically resize to
	// fill a variety of slot sizes.
	Attributes []string `json:"attributes,omitempty"`

	// ClickThroughUrls: The set of destination URLs for the creative.
	ClickThroughUrls []string `json:"clickThroughUrls,omitempty"`

	// Corrections: @OutputOnly Shows any corrections that were applied to
	// this creative.
	Corrections []*Correction `json:"corrections,omitempty"`

	// CreativeId: The buyer-defined creative ID of this creative.
	// Can be used to filter the response of the
	// creatives.list
	// method.
	CreativeId string `json:"creativeId,omitempty"`

	// DealsStatus: @OutputOnly The top-level deals status of this
	// creative.
	// If disapproved, an entry for 'auctionType=DIRECT_DEALS' (or 'ALL')
	// in
	// serving_restrictions will also exist. Note
	// that this may be nuanced with other contextual restrictions, in which
	// case,
	// it may be preferable to read from serving_restrictions directly.
	// Can be used to filter the response of the
	// creatives.list
	// method.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - The status is unknown.
	//   "NOT_CHECKED" - The creative has not been checked.
	//   "CONDITIONALLY_APPROVED" - The creative has been conditionally
	// approved.
	// See serving_restrictions for details.
	//   "APPROVED" - The creative has been approved.
	//   "DISAPPROVED" - The creative has been disapproved.
	DealsStatus string `json:"dealsStatus,omitempty"`

	// DetectedAdvertiserIds: @OutputOnly Detected advertiser IDs, if any.
	DetectedAdvertiserIds googleapi.Int64s `json:"detectedAdvertiserIds,omitempty"`

	// DetectedDomains: @OutputOnly
	// The detected domains for this creative.
	DetectedDomains []string `json:"detectedDomains,omitempty"`

	// DetectedLanguages: @OutputOnly
	// The detected languages for this creative. The order is arbitrary. The
	// codes
	// are 2 or 5 characters and are documented
	// at
	// https://developers.google.com/adwords/api/docs/appendix/languagecod
	// es.
	DetectedLanguages []string `json:"detectedLanguages,omitempty"`

	// DetectedProductCategories: @OutputOnly Detected product categories,
	// if any.
	// See the ad-product-categories.txt file in the technical
	// documentation
	// for a list of IDs.
	DetectedProductCategories []int64 `json:"detectedProductCategories,omitempty"`

	// DetectedSensitiveCategories: @OutputOnly Detected sensitive
	// categories, if any.
	// See the ad-sensitive-categories.txt file in the technical
	// documentation for
	// a list of IDs. You should use these IDs along with
	// the
	// excluded-sensitive-category field in the bid request to filter your
	// bids.
	DetectedSensitiveCategories []int64 `json:"detectedSensitiveCategories,omitempty"`

	// FilteringStats: @OutputOnly The filtering stats for this creative.
	FilteringStats *FilteringStats `json:"filteringStats,omitempty"`

	// Html: An HTML creative.
	Html *HtmlContent `json:"html,omitempty"`

	// ImpressionTrackingUrls: The set of URLs to be called to record an
	// impression.
	ImpressionTrackingUrls []string `json:"impressionTrackingUrls,omitempty"`

	// Native: A native creative.
	Native *NativeContent `json:"native,omitempty"`

	// OpenAuctionStatus: @OutputOnly The top-level open auction status of
	// this creative.
	// If disapproved, an entry for 'auctionType = OPEN_AUCTION' (or 'ALL')
	// in
	// serving_restrictions will also exist. Note
	// that this may be nuanced with other contextual restrictions, in which
	// case,
	// it may be preferable to read from serving_restrictions directly.
	// Can be used to filter the response of the
	// creatives.list
	// method.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - The status is unknown.
	//   "NOT_CHECKED" - The creative has not been checked.
	//   "CONDITIONALLY_APPROVED" - The creative has been conditionally
	// approved.
	// See serving_restrictions for details.
	//   "APPROVED" - The creative has been approved.
	//   "DISAPPROVED" - The creative has been disapproved.
	OpenAuctionStatus string `json:"openAuctionStatus,omitempty"`

	// RestrictedCategories: All restricted categories for the ads that may
	// be shown from this creative.
	//
	// Possible values:
	//   "NO_RESTRICTED_CATEGORIES" - The ad has no restricted categories
	//   "ALCOHOL" - The alcohol restricted category.
	RestrictedCategories []string `json:"restrictedCategories,omitempty"`

	// ServingRestrictions: @OutputOnly The granular status of this ad in
	// specific contexts.
	// A context here relates to where something ultimately serves (for
	// example,
	// a physical location, a platform, an HTTPS vs HTTP request, or the
	// type
	// of auction).
	ServingRestrictions []*ServingRestriction `json:"servingRestrictions,omitempty"`

	// VendorIds: All vendor IDs for the ads that may be shown from this
	// creative.
	// See
	// https://storage.googleapis.com/adx-rtb-dictionaries/vendors.txt
	// for possible values.
	VendorIds []int64 `json:"vendorIds,omitempty"`

	// Version: @OutputOnly The version of this creative.
	Version int64 `json:"version,omitempty"`

	// Video: A video creative.
	Video *VideoContent `json:"video,omitempty"`

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

func (s *Creative) MarshalJSON() ([]byte, error) {
	type noMethod Creative
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreativeDealAssociation: The association between a creative and a
// deal.
type CreativeDealAssociation struct {
	// AccountId: The account the creative belongs to.
	AccountId string `json:"accountId,omitempty"`

	// CreativeId: The ID of the creative associated with the deal.
	CreativeId string `json:"creativeId,omitempty"`

	// DealsId: The externalDealId for the deal associated with the
	// creative.
	DealsId string `json:"dealsId,omitempty"`

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

func (s *CreativeDealAssociation) MarshalJSON() ([]byte, error) {
	type noMethod CreativeDealAssociation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreativeStatusRow: The number of bids with the specified dimension
// values that did not win the
// auction (either were filtered pre-auction or lost the auction), as
// described
// by the specified creative status.
type CreativeStatusRow struct {
	// BidCount: The number of bids with the specified status.
	BidCount *MetricValue `json:"bidCount,omitempty"`

	// CreativeStatusId: The ID of the creative status.
	// See
	// [creative-status-codes](https://developers.google.com/ad-exchange/rtb/
	// downloads/creative-status-codes).
	CreativeStatusId int64 `json:"creativeStatusId,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BidCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BidCount") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreativeStatusRow) MarshalJSON() ([]byte, error) {
	type noMethod CreativeStatusRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Date: Represents a whole calendar date, e.g. date of birth. The time
// of day and
// time zone are either specified elsewhere or are not significant. The
// date
// is relative to the Proleptic Gregorian Calendar. The day may be 0
// to
// represent a year and month where the day is not significant, e.g.
// credit card
// expiration date. The year may be 0 to represent a month and day
// independent
// of year, e.g. anniversary date. Related types are
// google.type.TimeOfDay
// and `google.protobuf.Timestamp`.
type Date struct {
	// Day: Day of month. Must be from 1 to 31 and valid for the year and
	// month, or 0
	// if specifying a year/month where the day is not significant.
	Day int64 `json:"day,omitempty"`

	// Month: Month of year. Must be from 1 to 12.
	Month int64 `json:"month,omitempty"`

	// Year: Year of date. Must be from 1 to 9999, or 0 if specifying a date
	// without
	// a year.
	Year int64 `json:"year,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Day") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Day") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Date) MarshalJSON() ([]byte, error) {
	type noMethod Date
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Disapproval: @OutputOnly The reason and details for a disapproval.
type Disapproval struct {
	// Details: Additional details about the reason for disapproval.
	Details []string `json:"details,omitempty"`

	// Reason: The categorized reason for disapproval.
	//
	// Possible values:
	//   "LENGTH_OF_IMAGE_ANIMATION" - The length of the image animation is
	// longer than allowed.
	//   "BROKEN_URL" - The click through URL doesn't work properly.
	//   "MEDIA_NOT_FUNCTIONAL" - Something is wrong with the creative
	// itself.
	//   "INVALID_FOURTH_PARTY_CALL" - The ad makes a fourth party call to
	// an unapproved vendor.
	//   "INCORRECT_REMARKETING_DECLARATION" - The ad targets consumers
	// using remarketing lists and/or collects
	// data for subsequent use in retargeting, but does not correctly
	// declare
	// that use.
	//   "LANDING_PAGE_ERROR" - Clicking on the ad leads to an error page.
	//   "AD_SIZE_DOES_NOT_MATCH_AD_SLOT" - The ad size when rendered does
	// not match the declaration.
	//   "NO_BORDER" - Ads with a white background require a border, which
	// was missing.
	//   "FOURTH_PARTY_BROWSER_COOKIES" - The creative attempts to set
	// cookies from a fourth party that is not
	// certified.
	//   "LSO_OBJECTS" - The creative sets an LSO object.
	//   "BLANK_CREATIVE" - The ad serves a blank.
	//   "DESTINATION_URLS_UNDECLARED" - The ad uses rotation, but not all
	// destination URLs were declared.
	//   "PROBLEM_WITH_CLICK_MACRO" - There is a problem with the way the
	// click macro is used.
	//   "INCORRECT_AD_TECHNOLOGY_DECLARATION" - The ad technology
	// declaration is not accurate.
	//   "INCORRECT_DESTINATION_URL_DECLARATION" - The actual destination
	// URL does not match the declared destination URL.
	//   "EXPANDABLE_INCORRECT_DIRECTION" - The declared expanding direction
	// does not match the actual direction.
	//   "EXPANDABLE_DIRECTION_NOT_SUPPORTED" - The ad does not expand in a
	// supported direction.
	//   "EXPANDABLE_INVALID_VENDOR" - The ad uses an expandable vendor that
	// is not supported.
	//   "EXPANDABLE_FUNCTIONALITY" - There was an issue with the expandable
	// ad.
	//   "VIDEO_INVALID_VENDOR" - The ad uses a video vendor that is not
	// supported.
	//   "VIDEO_UNSUPPORTED_LENGTH" - The length of the video ad is not
	// supported.
	//   "VIDEO_UNSUPPORTED_FORMAT" - The format of the video ad is not
	// supported.
	//   "VIDEO_FUNCTIONALITY" - There was an issue with the video ad.
	//   "LANDING_PAGE_DISABLED" - The landing page does not conform to Ad
	// Exchange policy.
	//   "MALWARE_SUSPECTED" - The ad or the landing page may contain
	// malware.
	//   "ADULT_IMAGE_OR_VIDEO" - The ad contains adult images or video
	// content.
	//   "INACCURATE_AD_TEXT" - The ad contains text that is unclear or
	// inaccurate.
	//   "COUNTERFEIT_DESIGNER_GOODS" - The ad promotes counterfeit designer
	// goods.
	//   "POP_UP" - The ad causes a popup window to appear.
	//   "INVALID_RTB_PROTOCOL_USAGE" - The creative does not follow
	// policies set for the RTB protocol.
	//   "RAW_IP_ADDRESS_IN_SNIPPET" - The ad contains a URL that uses a
	// numeric IP address for the domain.
	//   "UNACCEPTABLE_CONTENT_SOFTWARE" - The ad or landing page contains
	// unacceptable content because it initiated
	// a software or executable download.
	//   "UNAUTHORIZED_COOKIE_ON_GOOGLE_DOMAIN" - The ad set an unauthorized
	// cookie on a Google domain.
	//   "UNDECLARED_FLASH_OBJECTS" - Flash content found when no flash was
	// declared.
	//   "INVALID_SSL_DECLARATION" - SSL support declared but not working
	// correctly.
	//   "DIRECT_DOWNLOAD_IN_AD" - Rich Media - Direct Download in Ad (ex.
	// PDF download).
	//   "MAXIMUM_DOWNLOAD_SIZE_EXCEEDED" - Maximum download size exceeded.
	//   "DESTINATION_URL_SITE_NOT_CRAWLABLE" - Bad Destination URL: Site
	// Not Crawlable.
	//   "BAD_URL_LEGAL_DISAPPROVAL" - Bad URL: Legal disapproval.
	//   "PHARMA_GAMBLING_ALCOHOL_NOT_ALLOWED" - Pharmaceuticals, Gambling,
	// Alcohol not allowed and at least one was
	// detected.
	//   "DYNAMIC_DNS_AT_DESTINATION_URL" - Dynamic DNS at Destination URL.
	//   "POOR_IMAGE_OR_VIDEO_QUALITY" - Poor Image / Video Quality.
	//   "UNACCEPTABLE_IMAGE_CONTENT" - For example, Image Trick to Click.
	//   "INCORRECT_IMAGE_LAYOUT" - Incorrect Image Layout.
	//   "IRRELEVANT_IMAGE_OR_VIDEO" - Irrelevant Image / Video.
	//   "DESTINATION_SITE_DOES_NOT_ALLOW_GOING_BACK" - Broken back button.
	//   "MISLEADING_CLAIMS_IN_AD" - Misleading/Inaccurate claims in ads.
	//   "RESTRICTED_PRODUCTS" - Restricted Products.
	//   "UNACCEPTABLE_CONTENT" - Unacceptable content. For example,
	// malware.
	//   "AUTOMATED_AD_CLICKING" - The ad automatically redirects to the
	// destination site without a click,
	// or reports a click when none were made.
	//   "INVALID_URL_PROTOCOL" - The ad uses URL protocols that do not
	// exist or are not allowed on AdX.
	//   "UNDECLARED_RESTRICTED_CONTENT" - Restricted content (for example,
	// alcohol) was found in the ad but not
	// declared.
	//   "INVALID_REMARKETING_LIST_USAGE" - Violation of the remarketing
	// list policy.
	//   "DESTINATION_SITE_NOT_CRAWLABLE_ROBOTS_TXT" - The destination
	// site's robot.txt file prevents it from being crawled.
	//   "CLICK_TO_DOWNLOAD_NOT_AN_APP" - Click to download must link to an
	// app.
	//   "INACCURATE_REVIEW_EXTENSION" - A review extension must be an
	// accurate review.
	//   "SEXUALLY_EXPLICIT_CONTENT" - Sexually explicit content.
	//   "GAINING_AN_UNFAIR_ADVANTAGE" - The ad tries to gain an unfair
	// traffic advantage.
	//   "GAMING_THE_GOOGLE_NETWORK" - The ad tries to circumvent Google's
	// advertising systems.
	//   "DANGEROUS_PRODUCTS_KNIVES" - The ad promotes dangerous knives.
	//   "DANGEROUS_PRODUCTS_EXPLOSIVES" - The ad promotes explosives.
	//   "DANGEROUS_PRODUCTS_GUNS" - The ad promotes guns & parts.
	//   "DANGEROUS_PRODUCTS_DRUGS" - The ad promotes recreational
	// drugs/services & related equipment.
	//   "DANGEROUS_PRODUCTS_TOBACCO" - The ad promotes tobacco
	// products/services & related equipment.
	//   "DANGEROUS_PRODUCTS_WEAPONS" - The ad promotes weapons.
	//   "UNCLEAR_OR_IRRELEVANT_AD" - The ad is unclear or irrelevant to the
	// destination site.
	//   "PROFESSIONAL_STANDARDS" - The ad does not meet professional
	// standards.
	//   "DYSFUNCTIONAL_PROMOTION" - The promotion is unnecessarily
	// difficult to navigate.
	//   "INVALID_INTEREST_BASED_AD" - Violation of Google's policy for
	// interest-based ads.
	//   "MISUSE_OF_PERSONAL_INFORMATION" - Misuse of personal information.
	//   "OMISSION_OF_RELEVANT_INFORMATION" - Omission of relevant
	// information.
	//   "UNAVAILABLE_PROMOTIONS" - Unavailable promotions.
	//   "MISLEADING_PROMOTIONS" - Misleading or unrealistic promotions.
	//   "INAPPROPRIATE_CONTENT" - Offensive or inappropriate content.
	//   "SENSITIVE_EVENTS" - Capitalizing on sensitive events.
	//   "SHOCKING_CONTENT" - Shocking content.
	//   "ENABLING_DISHONEST_BEHAVIOR" - Products & Services that enable
	// dishonest behavior.
	//   "TECHNICAL_REQUIREMENTS" - The ad does not meet technical
	// requirements.
	//   "RESTRICTED_POLITICAL_CONTENT" - Restricted political content.
	//   "UNSUPPORTED_CONTENT" - Unsupported content.
	//   "INVALID_BIDDING_METHOD" - Invalid bidding method.
	//   "VIDEO_TOO_LONG" - Video length exceeds limits.
	//   "VIOLATES_JAPANESE_PHARMACY_LAW" - Unacceptable content: Japanese
	// healthcare.
	//   "UNACCREDITED_PET_PHARMACY" - Online pharmacy ID required.
	//   "ABORTION" - Unacceptable content: Abortion.
	//   "CONTRACEPTIVES" - Unacceptable content: Birth control.
	//   "NEED_CERTIFICATES_TO_ADVERTISE_IN_CHINA" - Restricted in China.
	//   "KCDSP_REGISTRATION" - Unacceptable content: Korean healthcare.
	//   "NOT_FAMILY_SAFE" - Non-family safe or adult content.
	//   "CLINICAL_TRIAL_RECRUITMENT" - Clinical trial recruitment.
	//   "MAXIMUM_NUMBER_OF_HTTP_CALLS_EXCEEDED" - Maximum number of HTTP
	// calls exceeded.
	//   "MAXIMUM_NUMBER_OF_COOKIES_EXCEEDED" - Maximum number of cookies
	// exceeded.
	//   "PERSONAL_LOANS" - Financial service ad does not adhere to
	// specifications.
	//   "UNSUPPORTED_FLASH_CONTENT" - Flash content was found in an
	// unsupported context.
	Reason string `json:"reason,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Details") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Disapproval) MarshalJSON() ([]byte, error) {
	type noMethod Disapproval
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated
// empty messages in your APIs. A typical example is to use it as the
// request
// or the response type of an API method. For instance:
//
//     service Foo {
//       rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
//     }
//
// The JSON representation for `Empty` is empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// FilterSet: A set of filters that is applied to a request for
// data.
// Within a filter set, an AND operation is performed across the
// filters
// represented by each field. An OR operation is performed across the
// filters
// represented by the multiple values of a repeated field.
// E.g.
// "format=VIDEO AND deal_id=12 AND (seller_network_id=34
// OR
// seller_network_id=56)"
type FilterSet struct {
	// AbsoluteDateRange: An absolute date range, defined by a start date
	// and an end date.
	// Interpreted relative to Pacific time zone.
	AbsoluteDateRange *AbsoluteDateRange `json:"absoluteDateRange,omitempty"`

	// BuyerAccountId: The ID of the buyer account on which to filter;
	// optional.
	BuyerAccountId int64 `json:"buyerAccountId,omitempty,string"`

	// CreativeId: The ID of the creative on which to filter; optional.
	CreativeId string `json:"creativeId,omitempty"`

	// DealId: The ID of the deal on which to filter; optional.
	DealId int64 `json:"dealId,omitempty,string"`

	// Environment: The environment on which to filter; optional.
	//
	// Possible values:
	//   "ENVIRONMENT_UNSPECIFIED" - A placeholder for an undefined
	// environment; indicates that no environment
	// filter will be applied.
	//   "WEB" - The ad impression appears on the web.
	//   "APP" - The ad impression appears in an app.
	Environment string `json:"environment,omitempty"`

	// FilterSetId: The ID of the filter set; unique within the account of
	// the filter set
	// owner.
	// The value of this field is ignored in create operations.
	FilterSetId int64 `json:"filterSetId,omitempty,string"`

	// Format: The format on which to filter; optional.
	//
	// Possible values:
	//   "FORMAT_UNSPECIFIED" - A placeholder for an undefined format;
	// indicates that no format filter
	// will be applied.
	//   "DISPLAY" - The ad impression is display format (i.e. an image).
	//   "VIDEO" - The ad impression is video format.
	Format string `json:"format,omitempty"`

	// OwnerAccountId: The account ID of the buyer who owns this filter
	// set.
	// The value of this field is ignored in create operations.
	OwnerAccountId int64 `json:"ownerAccountId,omitempty,string"`

	// Platforms: The list of platforms on which to filter; may be empty.
	// The filters
	// represented by multiple platforms are ORed together (i.e. if
	// non-empty,
	// results must match any one of the platforms).
	//
	// Possible values:
	//   "PLATFORM_UNSPECIFIED" - A placeholder for an undefined platform;
	// indicates that no platform
	// filter will be applied.
	//   "DESKTOP" - The ad impression appears on a desktop.
	//   "TABLET" - The ad impression appears on a tablet.
	//   "MOBILE" - The ad impression appears on a mobile device.
	Platforms []string `json:"platforms,omitempty"`

	// RealtimeTimeRange: An open-ended realtime time range, defined by the
	// aggregation start
	// timestamp.
	RealtimeTimeRange *RealtimeTimeRange `json:"realtimeTimeRange,omitempty"`

	// RelativeDateRange: A relative date range, defined by an offset from
	// today and a duration.
	// Interpreted relative to Pacific time zone.
	RelativeDateRange *RelativeDateRange `json:"relativeDateRange,omitempty"`

	// SellerNetworkIds: The list of IDs of the seller (publisher) networks
	// on which to filter;
	// may be empty. The filters represented by multiple seller network IDs
	// are
	// ORed together (i.e. if non-empty, results must match any one of
	// the
	// publisher networks).
	// See
	// [seller-network-ids](https://developers.google.com/ad-exchange/rtb/dow
	// nloads/seller-network-ids)
	// file for the set of existing seller network IDs.
	SellerNetworkIds []int64 `json:"sellerNetworkIds,omitempty"`

	// TimeSeriesGranularity: The granularity of time intervals if a time
	// series breakdown is desired;
	// optional.
	//
	// Possible values:
	//   "TIME_SERIES_GRANULARITY_UNSPECIFIED" - A placeholder for an
	// unspecified interval; no time series is applied.
	// All rows in response will contain data for the entire requested time
	// range.
	//   "HOURLY" - Indicates that data will be broken down by the hour.
	//   "DAILY" - Indicates that data will be broken down by the day.
	TimeSeriesGranularity string `json:"timeSeriesGranularity,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AbsoluteDateRange")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AbsoluteDateRange") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FilterSet) MarshalJSON() ([]byte, error) {
	type noMethod FilterSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FilteredBidCreativeRow: The number of filtered bids with the
// specified dimension values that have the
// specified creative.
type FilteredBidCreativeRow struct {
	// BidCount: The number of bids with the specified creative.
	BidCount *MetricValue `json:"bidCount,omitempty"`

	// CreativeId: The ID of the creative.
	CreativeId string `json:"creativeId,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BidCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BidCount") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FilteredBidCreativeRow) MarshalJSON() ([]byte, error) {
	type noMethod FilteredBidCreativeRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FilteredBidDetailRow: The number of filtered bids with the specified
// dimension values, among those
// filtered due to the requested filtering reason (i.e. creative
// status), that
// have the specified detail.
type FilteredBidDetailRow struct {
	// BidCount: The number of bids with the specified detail.
	BidCount *MetricValue `json:"bidCount,omitempty"`

	// DetailId: The ID of the detail. The associated value can be looked up
	// in the
	// dictionary file corresponding to the DetailType in the response
	// message.
	DetailId int64 `json:"detailId,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BidCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BidCount") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FilteredBidDetailRow) MarshalJSON() ([]byte, error) {
	type noMethod FilteredBidDetailRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FilteringStats: @OutputOnly Filtering reasons for this creative
// during a period of a single
// day (from midnight to midnight Pacific).
type FilteringStats struct {
	// Date: The day during which the data was collected.
	// The data is collected from 00:00:00 to 23:59:59 PT.
	// During switches from PST to PDT and back, the day may
	// contain 23 or 25 hours of data instead of the usual 24.
	Date *Date `json:"date,omitempty"`

	// Reasons: The set of filtering reasons for this date.
	Reasons []*Reason `json:"reasons,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Date") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Date") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FilteringStats) MarshalJSON() ([]byte, error) {
	type noMethod FilteringStats
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HtmlContent: HTML content for a creative.
type HtmlContent struct {
	// Height: The height of the HTML snippet in pixels.
	Height int64 `json:"height,omitempty"`

	// Snippet: The HTML snippet that displays the ad when inserted in the
	// web page.
	Snippet string `json:"snippet,omitempty"`

	// Width: The width of the HTML snippet in pixels.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Height") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HtmlContent) MarshalJSON() ([]byte, error) {
	type noMethod HtmlContent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Image: An image resource. You may provide a larger image than was
// requested,
// so long as the aspect ratio is preserved.
type Image struct {
	// Height: Image height in pixels.
	Height int64 `json:"height,omitempty"`

	// Url: The URL of the image.
	Url string `json:"url,omitempty"`

	// Width: Image width in pixels.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Height") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Image) MarshalJSON() ([]byte, error) {
	type noMethod Image
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImpressionMetricsRow: The set of metrics that are measured in numbers
// of impressions, representing
// how many impressions with the specified dimension values were
// considered
// eligible at each stage of the bidding funnel.
type ImpressionMetricsRow struct {
	// AvailableImpressions: The number of impressions available to the
	// buyer on Ad Exchange.
	// In some cases this value may be unavailable.
	AvailableImpressions *MetricValue `json:"availableImpressions,omitempty"`

	// BidRequests: The number of impressions for which Ad Exchange sent the
	// buyer a bid
	// request.
	BidRequests *MetricValue `json:"bidRequests,omitempty"`

	// InventoryMatches: The number of impressions that match the buyer's
	// inventory pretargeting.
	InventoryMatches *MetricValue `json:"inventoryMatches,omitempty"`

	// ResponsesWithBids: The number of impressions for which Ad Exchange
	// received a response from
	// the buyer that contained at least one applicable bid.
	ResponsesWithBids *MetricValue `json:"responsesWithBids,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// SuccessfulResponses: The number of impressions for which the buyer
	// successfully sent a response
	// to Ad Exchange.
	SuccessfulResponses *MetricValue `json:"successfulResponses,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AvailableImpressions") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AvailableImpressions") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ImpressionMetricsRow) MarshalJSON() ([]byte, error) {
	type noMethod ImpressionMetricsRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListBidMetricsResponse: Response message for listing the metrics that
// are measured in number of bids.
type ListBidMetricsResponse struct {
	// BidMetricsRows: List of rows, each containing a set of bid metrics.
	BidMetricsRows []*BidMetricsRow `json:"bidMetricsRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListBidMetricsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.bidMetrics.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BidMetricsRows") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BidMetricsRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListBidMetricsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListBidMetricsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListBidResponseErrorsResponse: Response message for listing all
// reasons that bid responses resulted in an
// error.
type ListBidResponseErrorsResponse struct {
	// CalloutStatusRows: List of rows, with counts of bid responses
	// aggregated by callout status.
	CalloutStatusRows []*CalloutStatusRow `json:"calloutStatusRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListBidResponseErrorsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.bidResponseErrors.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CalloutStatusRows")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CalloutStatusRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListBidResponseErrorsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListBidResponseErrorsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListBidResponsesWithoutBidsResponse: Response message for listing all
// reasons that bid responses were considered
// to have no applicable bids.
type ListBidResponsesWithoutBidsResponse struct {
	// BidResponseWithoutBidsStatusRows: List of rows, with counts of bid
	// responses without bids aggregated by
	// status.
	BidResponseWithoutBidsStatusRows []*BidResponseWithoutBidsStatusRow `json:"bidResponseWithoutBidsStatusRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in
	// the
	// ListBidResponsesWithoutBidsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.bidResponsesWithoutBids.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "BidResponseWithoutBidsStatusRows") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "BidResponseWithoutBidsStatusRows") to include in API requests with
	// the JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListBidResponsesWithoutBidsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListBidResponsesWithoutBidsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListClientUserInvitationsResponse struct {
	// Invitations: The returned list of client users.
	Invitations []*ClientUserInvitation `json:"invitations,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in
	// the
	// ListClientUserInvitationsRequest.pageToken
	// field in the subsequent call to the
	// clients.invitations.list
	// method to retrieve the next
	// page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Invitations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Invitations") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListClientUserInvitationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListClientUserInvitationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListClientUsersResponse struct {
	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListClientUsersRequest.pageToken
	// field in the subsequent call to the
	// clients.invitations.list
	// method to retrieve the next
	// page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Users: The returned list of client users.
	Users []*ClientUser `json:"users,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListClientUsersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListClientUsersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListClientsResponse struct {
	// Clients: The returned list of clients.
	Clients []*Client `json:"clients,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListClientsRequest.pageToken
	// field in the subsequent call to the
	// accounts.clients.list method
	// to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Clients") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Clients") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListClientsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListClientsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListCreativeStatusBreakdownByCreativeResponse: Response message for
// listing all creatives associated with a given filtered
// bid reason.
type ListCreativeStatusBreakdownByCreativeResponse struct {
	// FilteredBidCreativeRows: List of rows, with counts of bids with a
	// given creative status aggregated
	// by creative.
	FilteredBidCreativeRows []*FilteredBidCreativeRow `json:"filteredBidCreativeRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in
	// the
	// ListCreativeStatusBreakdownByCreativeRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.filteredBids.creatives.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "FilteredBidCreativeRows") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FilteredBidCreativeRows")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListCreativeStatusBreakdownByCreativeResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCreativeStatusBreakdownByCreativeResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListCreativeStatusBreakdownByDetailResponse: Response message for
// listing all details associated with a given filtered bid
// reason.
type ListCreativeStatusBreakdownByDetailResponse struct {
	// DetailType: The type of detail that the detail IDs represent.
	//
	// Possible values:
	//   "DETAIL_TYPE_UNSPECIFIED" - A placeholder for an undefined
	// status.
	// This value will never be returned in responses.
	//   "CREATIVE_ATTRIBUTE" - Indicates that the detail ID refers to a
	// creative attribute;
	// see
	// [publisher-excludable-creative-attributes](https://developers.goog
	// le.com/ad-exchange/rtb/downloads/publisher-excludable-creative-attribu
	// tes).
	//   "VENDOR" - Indicates that the detail ID refers to a vendor;
	// see
	// [vendors](https://developers.google.com/ad-exchange/rtb/downloads/
	// vendors).
	//   "SENSITIVE_CATEGORY" - Indicates that the detail ID refers to a
	// sensitive category;
	// see
	// [ad-sensitive-categories](https://developers.google.com/ad-exchang
	// e/rtb/downloads/ad-sensitive-categories).
	//   "PRODUCT_CATEGORY" - Indicates that the detail ID refers to a
	// product category;
	// see
	// [ad-product-categories](https://developers.google.com/ad-exchange/
	// rtb/downloads/ad-product-categories).
	//   "DISAPPROVAL_REASON" - Indicates that the detail ID refers to a
	// disapproval reason; see
	// DisapprovalReason enum in
	// [snippet-status-report-proto](https://developers.google.com/ad-exchang
	// e/rtb/downloads/snippet-status-report-proto).
	DetailType string `json:"detailType,omitempty"`

	// FilteredBidDetailRows: List of rows, with counts of bids with a given
	// creative status aggregated
	// by detail.
	FilteredBidDetailRows []*FilteredBidDetailRow `json:"filteredBidDetailRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in
	// the
	// ListCreativeStatusBreakdownByDetailRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.filteredBids.details.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DetailType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DetailType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListCreativeStatusBreakdownByDetailResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCreativeStatusBreakdownByDetailResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListCreativesResponse: A response for listing creatives.
type ListCreativesResponse struct {
	// Creatives: The list of creatives.
	Creatives []*Creative `json:"creatives,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListCreativesRequest.page_token
	// field in the subsequent call to `ListCreatives` method to retrieve
	// the next
	// page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Creatives") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Creatives") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListCreativesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCreativesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListDealAssociationsResponse: A response for listing creative and
// deal associations
type ListDealAssociationsResponse struct {
	// Associations: The list of associations.
	Associations []*CreativeDealAssociation `json:"associations,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListDealAssociationsRequest.page_token
	// field in the subsequent call to 'ListDealAssociation' method to
	// retrieve
	// the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Associations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Associations") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListDealAssociationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListDealAssociationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListFilterSetsResponse: Response message for listing filter sets.
type ListFilterSetsResponse struct {
	// FilterSets: The filter sets belonging to the buyer.
	FilterSets []*FilterSet `json:"filterSets,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListFilterSetsRequest.pageToken
	// field in the subsequent call to the
	// accounts.filterSets.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "FilterSets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FilterSets") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListFilterSetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListFilterSetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListFilteredBidRequestsResponse: Response message for listing all
// reasons that bid requests were filtered and
// not sent to the buyer.
type ListFilteredBidRequestsResponse struct {
	// CalloutStatusRows: List of rows, with counts of filtered bid requests
	// aggregated by callout
	// status.
	CalloutStatusRows []*CalloutStatusRow `json:"calloutStatusRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListFilteredBidRequestsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.filteredBidRequests.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CalloutStatusRows")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CalloutStatusRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListFilteredBidRequestsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListFilteredBidRequestsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListFilteredBidsResponse: Response message for listing all reasons
// that bids were filtered from the
// auction.
type ListFilteredBidsResponse struct {
	// CreativeStatusRows: List of rows, with counts of filtered bids
	// aggregated by filtering reason
	// (i.e. creative status).
	CreativeStatusRows []*CreativeStatusRow `json:"creativeStatusRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListFilteredBidsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.filteredBids.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreativeStatusRows")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreativeStatusRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListFilteredBidsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListFilteredBidsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListImpressionMetricsResponse: Response message for listing the
// metrics that are measured in number of
// impressions.
type ListImpressionMetricsResponse struct {
	// ImpressionMetricsRows: List of rows, each containing a set of
	// impression metrics.
	ImpressionMetricsRows []*ImpressionMetricsRow `json:"impressionMetricsRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListImpressionMetricsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.impressionMetrics.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "ImpressionMetricsRows") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ImpressionMetricsRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListImpressionMetricsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListImpressionMetricsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListLosingBidsResponse: Response message for listing all reasons that
// bids lost in the auction.
type ListLosingBidsResponse struct {
	// CreativeStatusRows: List of rows, with counts of losing bids
	// aggregated by loss reason (i.e.
	// creative status).
	CreativeStatusRows []*CreativeStatusRow `json:"creativeStatusRows,omitempty"`

	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in the
	// ListLosingBidsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.losingBids.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreativeStatusRows")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreativeStatusRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListLosingBidsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLosingBidsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListNonBillableWinningBidsResponse: Response message for listing all
// reasons for which a buyer was not billed for
// a winning bid.
type ListNonBillableWinningBidsResponse struct {
	// NextPageToken: A token to retrieve the next page of results.
	// Pass this value in
	// the
	// ListNonBillableWinningBidsRequest.pageToken
	// field in the subsequent call to
	// the
	// accounts.filterSets.nonBillableWinningBids.list
	// method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// NonBillableWinningBidStatusRows: List of rows, with counts of bids
	// not billed aggregated by reason.
	NonBillableWinningBidStatusRows []*NonBillableWinningBidStatusRow `json:"nonBillableWinningBidStatusRows,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListNonBillableWinningBidsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListNonBillableWinningBidsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LocationContext: @OutputOnly The Geo criteria the restriction applies
// to.
type LocationContext struct {
	// GeoCriteriaIds: IDs representing the geo location for this
	// context.
	// Please refer to
	// the
	// [geo-table.csv](https://storage.googleapis.com/adx-rtb-dictionarie
	// s/geo-table.csv)
	// file for different geo criteria IDs.
	GeoCriteriaIds []int64 `json:"geoCriteriaIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "GeoCriteriaIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GeoCriteriaIds") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *LocationContext) MarshalJSON() ([]byte, error) {
	type noMethod LocationContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricValue: A metric value, with an expected value and a variance;
// represents a count
// that may be either exact or estimated (i.e. when sampled).
type MetricValue struct {
	// Value: The expected value of the metric.
	Value int64 `json:"value,omitempty,string"`

	// Variance: The variance (i.e. square of the standard deviation) of the
	// metric value.
	// If value is exact, variance is 0.
	// Can be used to calculate margin of error as a percentage of value,
	// using
	// the following formula, where Z is the standard constant that depends
	// on the
	// desired size of the confidence interval (e.g. for 90% confidence
	// interval,
	// use Z = 1.645):
	//
	//   marginOfError = 100 * Z * sqrt(variance) / value
	Variance int64 `json:"variance,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Value") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Value") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricValue) MarshalJSON() ([]byte, error) {
	type noMethod MetricValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NativeContent: Native content for a creative.
type NativeContent struct {
	// AdvertiserName: The name of the advertiser or sponsor, to be
	// displayed in the ad creative.
	AdvertiserName string `json:"advertiserName,omitempty"`

	// AppIcon: The app icon, for app download ads.
	AppIcon *Image `json:"appIcon,omitempty"`

	// Body: A long description of the ad.
	Body string `json:"body,omitempty"`

	// CallToAction: A label for the button that the user is supposed to
	// click.
	CallToAction string `json:"callToAction,omitempty"`

	// ClickLinkUrl: The URL that the browser/SDK will load when the user
	// clicks the ad.
	ClickLinkUrl string `json:"clickLinkUrl,omitempty"`

	// ClickTrackingUrl: The URL to use for click tracking.
	ClickTrackingUrl string `json:"clickTrackingUrl,omitempty"`

	// Headline: A short title for the ad.
	Headline string `json:"headline,omitempty"`

	// Image: A large image.
	Image *Image `json:"image,omitempty"`

	// Logo: A smaller image, for the advertiser's logo.
	Logo *Image `json:"logo,omitempty"`

	// PriceDisplayText: The price of the promoted app including currency
	// info.
	PriceDisplayText string `json:"priceDisplayText,omitempty"`

	// StarRating: The app rating in the app store. Must be in the range
	// [0-5].
	StarRating float64 `json:"starRating,omitempty"`

	// StoreUrl: The URL to the app store to purchase/download the promoted
	// app.
	StoreUrl string `json:"storeUrl,omitempty"`

	// VideoUrl: The URL to fetch a native video ad.
	VideoUrl string `json:"videoUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdvertiserName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdvertiserName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *NativeContent) MarshalJSON() ([]byte, error) {
	type noMethod NativeContent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *NativeContent) UnmarshalJSON(data []byte) error {
	type noMethod NativeContent
	var s1 struct {
		StarRating gensupport.JSONFloat64 `json:"starRating"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.StarRating = float64(s1.StarRating)
	return nil
}

// NonBillableWinningBidStatusRow: The number of winning bids with the
// specified dimension values for which the
// buyer was not billed, as described by the specified status.
type NonBillableWinningBidStatusRow struct {
	// BidCount: The number of bids with the specified status.
	BidCount *MetricValue `json:"bidCount,omitempty"`

	// RowDimensions: The values of all dimensions associated with metric
	// values in this row.
	RowDimensions *RowDimensions `json:"rowDimensions,omitempty"`

	// Status: The status specifying why the winning bids were not billed.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - A placeholder for an undefined status.
	// This value will never be returned in responses.
	//   "AD_NOT_RENDERED" - The buyer was not billed because the ad was not
	// rendered by the
	// publisher.
	//   "INVALID_IMPRESSION" - The buyer was not billed because the
	// impression won by the bid was
	// determined to be invalid.
	Status string `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BidCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BidCount") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NonBillableWinningBidStatusRow) MarshalJSON() ([]byte, error) {
	type noMethod NonBillableWinningBidStatusRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PlatformContext: @OutputOnly The type of platform the restriction
// applies to.
type PlatformContext struct {
	// Platforms: The platforms this restriction applies to.
	//
	// Possible values:
	//   "DESKTOP" - Desktop platform.
	//   "ANDROID" - Android platform.
	//   "IOS" - iOS platform.
	Platforms []string `json:"platforms,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Platforms") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Platforms") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PlatformContext) MarshalJSON() ([]byte, error) {
	type noMethod PlatformContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RealtimeTimeRange: An open-ended realtime time range specified by the
// start timestamp.
// For filter sets that specify a realtime time range RTB metrics
// continue to
// be aggregated throughout the lifetime of the filter set.
type RealtimeTimeRange struct {
	// StartTimestamp: The start timestamp of the real-time RTB metrics
	// aggregation.
	StartTimestamp string `json:"startTimestamp,omitempty"`

	// ForceSendFields is a list of field names (e.g. "StartTimestamp") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "StartTimestamp") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *RealtimeTimeRange) MarshalJSON() ([]byte, error) {
	type noMethod RealtimeTimeRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Reason: A specific filtering status and how many times it occurred.
type Reason struct {
	// Count: The number of times the creative was filtered for the status.
	// The
	// count is aggregated across all publishers on the exchange.
	Count int64 `json:"count,omitempty,string"`

	// Status: The filtering status code. Please refer to
	// the
	// [creative-status-codes.txt](https://storage.googleapis.com/adx-rtb
	// -dictionaries/creative-status-codes.txt)
	// file for different statuses.
	Status int64 `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Count") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Reason) MarshalJSON() ([]byte, error) {
	type noMethod Reason
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RelativeDateRange: A relative date range, specified by an offset and
// a duration.
// The supported range of dates begins 30 days before today and ends
// today.
// I.e. the limits for these values are:
// offset_days >= 0
// duration_days >= 1
// offset_days + duration_days <= 30
type RelativeDateRange struct {
	// DurationDays: The number of days in the requested date range. E.g.
	// for a range spanning
	// today, 1. For a range spanning the last 7 days, 7.
	DurationDays int64 `json:"durationDays,omitempty"`

	// OffsetDays: The end date of the filter set, specified as the number
	// of days before
	// today. E.g. for a range where the last date is today, 0.
	OffsetDays int64 `json:"offsetDays,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DurationDays") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DurationDays") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RelativeDateRange) MarshalJSON() ([]byte, error) {
	type noMethod RelativeDateRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RemoveDealAssociationRequest: A request for removing the association
// between a deal and a creative.
type RemoveDealAssociationRequest struct {
	// Association: The association between a creative and a deal that
	// should be removed.
	Association *CreativeDealAssociation `json:"association,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Association") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Association") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RemoveDealAssociationRequest) MarshalJSON() ([]byte, error) {
	type noMethod RemoveDealAssociationRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RowDimensions: A response may include multiple rows, breaking down
// along various dimensions.
// Encapsulates the values of all dimensions for a given row.
type RowDimensions struct {
	// TimeInterval: The time interval that this row represents.
	TimeInterval *TimeInterval `json:"timeInterval,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TimeInterval") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TimeInterval") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RowDimensions) MarshalJSON() ([]byte, error) {
	type noMethod RowDimensions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SecurityContext: @OutputOnly A security context.
type SecurityContext struct {
	// Securities: The security types in this context.
	//
	// Possible values:
	//   "INSECURE" - Matches impressions that require insecure
	// compatibility.
	//   "SSL" - Matches impressions that require SSL compatibility.
	Securities []string `json:"securities,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Securities") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Securities") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SecurityContext) MarshalJSON() ([]byte, error) {
	type noMethod SecurityContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ServingContext: The serving context for this restriction.
type ServingContext struct {
	// All: Matches all contexts.
	//
	// Possible values:
	//   "SIMPLE_CONTEXT" - A simple context.
	All string `json:"all,omitempty"`

	// AppType: Matches impressions for a particular app type.
	AppType *AppContext `json:"appType,omitempty"`

	// AuctionType: Matches impressions for a particular auction type.
	AuctionType *AuctionContext `json:"auctionType,omitempty"`

	// Location: Matches impressions coming from users *or* publishers in a
	// specific
	// location.
	Location *LocationContext `json:"location,omitempty"`

	// Platform: Matches impressions coming from a particular platform.
	Platform *PlatformContext `json:"platform,omitempty"`

	// SecurityType: Matches impressions for a particular security type.
	SecurityType *SecurityContext `json:"securityType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "All") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "All") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ServingContext) MarshalJSON() ([]byte, error) {
	type noMethod ServingContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ServingRestriction: @OutputOnly A representation of the status of an
// ad in a
// specific context. A context here relates to where something
// ultimately serves
// (for example, a user or publisher geo, a platform, an HTTPS vs HTTP
// request,
// or the type of auction).
type ServingRestriction struct {
	// Contexts: The contexts for the restriction.
	Contexts []*ServingContext `json:"contexts,omitempty"`

	// DisapprovalReasons: Any disapprovals bound to this restriction.
	// Only present if status=DISAPPROVED.
	// Can be used to filter the response of the
	// creatives.list
	// method.
	DisapprovalReasons []*Disapproval `json:"disapprovalReasons,omitempty"`

	// Status: The status of the creative in this context (for example, it
	// has been
	// explicitly disapproved or is pending review).
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - The status is not known.
	//   "DISAPPROVAL" - The ad was disapproved in this context.
	//   "PENDING_REVIEW" - The ad is pending review in this context.
	Status string `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Contexts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Contexts") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ServingRestriction) MarshalJSON() ([]byte, error) {
	type noMethod ServingRestriction
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// StopWatchingCreativeRequest: A request for stopping notifications for
// changes to creative Status.
type StopWatchingCreativeRequest struct {
}

// TimeInterval: An interval of time, with an absolute start and
// end.
// This is included in the response, for several reasons:
// 1) The request may have specified start or end times relative to the
// time the
//    request was sent; the response indicates the corresponding
// absolute time
//    interval.
// 2) The request may have specified an end time past the latest time
// for which
//    data was available (e.g. if requesting data for the today); the
// response
//    indicates the latest time for which data was actually returned.
// 3) The response data for a single request may be broken down into
// multiple
//    time intervals, if a time series was requested.
type TimeInterval struct {
	// EndTime: The timestamp marking the end of the range (exclusive) for
	// which data is
	// included.
	EndTime string `json:"endTime,omitempty"`

	// StartTime: The timestamp marking the start of the range (inclusive)
	// for which data is
	// included.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TimeInterval) MarshalJSON() ([]byte, error) {
	type noMethod TimeInterval
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// VideoContent: Video content for a creative.
type VideoContent struct {
	// VideoUrl: The URL to fetch a video ad.
	VideoUrl string `json:"videoUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "VideoUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "VideoUrl") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *VideoContent) MarshalJSON() ([]byte, error) {
	type noMethod VideoContent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WatchCreativeRequest: A request for watching changes to creative
// Status.
type WatchCreativeRequest struct {
	// Topic: The Pub/Sub topic to publish notifications to.
	// This topic must already exist and must give permission
	// to
	// ad-exchange-buyside-reports@google.com to write to the topic.
	// This should be the full resource name
	// in
	// "projects/{project_id}/topics/{topic_id}" format.
	Topic string `json:"topic,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Topic") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Topic") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WatchCreativeRequest) MarshalJSON() ([]byte, error) {
	type noMethod WatchCreativeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "adexchangebuyer2.accounts.clients.create":

type AccountsClientsCreateCall struct {
	s          *Service
	accountId  int64
	client     *Client
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new client buyer.
func (r *AccountsClientsService) Create(accountId int64, client *Client) *AccountsClientsCreateCall {
	c := &AccountsClientsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.client = client
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsCreateCall) Fields(s ...googleapi.Field) *AccountsClientsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsCreateCall) Context(ctx context.Context) *AccountsClientsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.client)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.create" call.
// Exactly one of *Client or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Client.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsClientsCreateCall) Do(opts ...googleapi.CallOption) (*Client, error) {
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
	ret := &Client{
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
	//   "description": "Creates a new client buyer.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.clients.create",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Unique numerical account ID for the buyer of which the client buyer\nis a customer; the sponsor buyer to create a client for. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients",
	//   "request": {
	//     "$ref": "Client"
	//   },
	//   "response": {
	//     "$ref": "Client"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.clients.get":

type AccountsClientsGetCall struct {
	s               *Service
	accountId       int64
	clientAccountId int64
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// Get: Gets a client buyer with a given client account ID.
func (r *AccountsClientsService) Get(accountId int64, clientAccountId int64) *AccountsClientsGetCall {
	c := &AccountsClientsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsGetCall) Fields(s ...googleapi.Field) *AccountsClientsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsClientsGetCall) IfNoneMatch(entityTag string) *AccountsClientsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsGetCall) Context(ctx context.Context) *AccountsClientsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": strconv.FormatInt(c.clientAccountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.get" call.
// Exactly one of *Client or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Client.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsClientsGetCall) Do(opts ...googleapi.CallOption) (*Client, error) {
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
	ret := &Client{
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
	//   "description": "Gets a client buyer with a given client account ID.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.clients.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Numerical account ID of the client's sponsor buyer. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "Numerical account ID of the client buyer to retrieve. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}",
	//   "response": {
	//     "$ref": "Client"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.clients.list":

type AccountsClientsListCall struct {
	s            *Service
	accountId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all the clients for the current sponsor buyer.
func (r *AccountsClientsService) List(accountId int64) *AccountsClientsListCall {
	c := &AccountsClientsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer clients than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsClientsListCall) PageSize(pageSize int64) *AccountsClientsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListClientsResponse.nextPageToken
// returned from the previous call to the
// accounts.clients.list method.
func (c *AccountsClientsListCall) PageToken(pageToken string) *AccountsClientsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsListCall) Fields(s ...googleapi.Field) *AccountsClientsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsClientsListCall) IfNoneMatch(entityTag string) *AccountsClientsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsListCall) Context(ctx context.Context) *AccountsClientsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.list" call.
// Exactly one of *ListClientsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListClientsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsClientsListCall) Do(opts ...googleapi.CallOption) (*ListClientsResponse, error) {
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
	ret := &ListClientsResponse{
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
	//   "description": "Lists all the clients for the current sponsor buyer.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.clients.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Unique numerical account ID of the sponsor buyer to list the clients for.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer clients than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListClientsResponse.nextPageToken\nreturned from the previous call to the\naccounts.clients.list method.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients",
	//   "response": {
	//     "$ref": "ListClientsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsClientsListCall) Pages(ctx context.Context, f func(*ListClientsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.clients.update":

type AccountsClientsUpdateCall struct {
	s               *Service
	accountId       int64
	clientAccountId int64
	client          *Client
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Update: Updates an existing client buyer.
func (r *AccountsClientsService) Update(accountId int64, clientAccountId int64, client *Client) *AccountsClientsUpdateCall {
	c := &AccountsClientsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	c.client = client
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsUpdateCall) Fields(s ...googleapi.Field) *AccountsClientsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsUpdateCall) Context(ctx context.Context) *AccountsClientsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.client)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": strconv.FormatInt(c.clientAccountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.update" call.
// Exactly one of *Client or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Client.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsClientsUpdateCall) Do(opts ...googleapi.CallOption) (*Client, error) {
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
	ret := &Client{
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
	//   "description": "Updates an existing client buyer.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}",
	//   "httpMethod": "PUT",
	//   "id": "adexchangebuyer2.accounts.clients.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Unique numerical account ID for the buyer of which the client buyer\nis a customer; the sponsor buyer to update a client for. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "Unique numerical account ID of the client to update. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}",
	//   "request": {
	//     "$ref": "Client"
	//   },
	//   "response": {
	//     "$ref": "Client"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.clients.invitations.create":

type AccountsClientsInvitationsCreateCall struct {
	s                    *Service
	accountId            int64
	clientAccountId      int64
	clientuserinvitation *ClientUserInvitation
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Create: Creates and sends out an email invitation to access
// an Ad Exchange client buyer account.
func (r *AccountsClientsInvitationsService) Create(accountId int64, clientAccountId int64, clientuserinvitation *ClientUserInvitation) *AccountsClientsInvitationsCreateCall {
	c := &AccountsClientsInvitationsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	c.clientuserinvitation = clientuserinvitation
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsInvitationsCreateCall) Fields(s ...googleapi.Field) *AccountsClientsInvitationsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsInvitationsCreateCall) Context(ctx context.Context) *AccountsClientsInvitationsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsInvitationsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsInvitationsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.clientuserinvitation)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": strconv.FormatInt(c.clientAccountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.invitations.create" call.
// Exactly one of *ClientUserInvitation or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ClientUserInvitation.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsClientsInvitationsCreateCall) Do(opts ...googleapi.CallOption) (*ClientUserInvitation, error) {
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
	ret := &ClientUserInvitation{
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
	//   "description": "Creates and sends out an email invitation to access\nan Ad Exchange client buyer account.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.clients.invitations.create",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Numerical account ID of the client's sponsor buyer. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "Numerical account ID of the client buyer that the user\nshould be associated with. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations",
	//   "request": {
	//     "$ref": "ClientUserInvitation"
	//   },
	//   "response": {
	//     "$ref": "ClientUserInvitation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.clients.invitations.get":

type AccountsClientsInvitationsGetCall struct {
	s               *Service
	accountId       int64
	clientAccountId int64
	invitationId    int64
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// Get: Retrieves an existing client user invitation.
func (r *AccountsClientsInvitationsService) Get(accountId int64, clientAccountId int64, invitationId int64) *AccountsClientsInvitationsGetCall {
	c := &AccountsClientsInvitationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	c.invitationId = invitationId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsInvitationsGetCall) Fields(s ...googleapi.Field) *AccountsClientsInvitationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsClientsInvitationsGetCall) IfNoneMatch(entityTag string) *AccountsClientsInvitationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsInvitationsGetCall) Context(ctx context.Context) *AccountsClientsInvitationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsInvitationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsInvitationsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations/{invitationId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": strconv.FormatInt(c.clientAccountId, 10),
		"invitationId":    strconv.FormatInt(c.invitationId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.invitations.get" call.
// Exactly one of *ClientUserInvitation or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ClientUserInvitation.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsClientsInvitationsGetCall) Do(opts ...googleapi.CallOption) (*ClientUserInvitation, error) {
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
	ret := &ClientUserInvitation{
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
	//   "description": "Retrieves an existing client user invitation.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations/{invitationId}",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.clients.invitations.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId",
	//     "invitationId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Numerical account ID of the client's sponsor buyer. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "Numerical account ID of the client buyer that the user invitation\nto be retrieved is associated with. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "invitationId": {
	//       "description": "Numerical identifier of the user invitation to retrieve. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations/{invitationId}",
	//   "response": {
	//     "$ref": "ClientUserInvitation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.clients.invitations.list":

type AccountsClientsInvitationsListCall struct {
	s               *Service
	accountId       int64
	clientAccountId string
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// List: Lists all the client users invitations for a client
// with a given account ID.
func (r *AccountsClientsInvitationsService) List(accountId int64, clientAccountId string) *AccountsClientsInvitationsListCall {
	c := &AccountsClientsInvitationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// Server may return fewer clients than requested.
// If unspecified, server will pick an appropriate default.
func (c *AccountsClientsInvitationsListCall) PageSize(pageSize int64) *AccountsClientsInvitationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListClientUserInvitationsResponse.nextPageToken
// returned from the previous call to
// the
// clients.invitations.list
// method.
func (c *AccountsClientsInvitationsListCall) PageToken(pageToken string) *AccountsClientsInvitationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsInvitationsListCall) Fields(s ...googleapi.Field) *AccountsClientsInvitationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsClientsInvitationsListCall) IfNoneMatch(entityTag string) *AccountsClientsInvitationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsInvitationsListCall) Context(ctx context.Context) *AccountsClientsInvitationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsInvitationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsInvitationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": c.clientAccountId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.invitations.list" call.
// Exactly one of *ListClientUserInvitationsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListClientUserInvitationsResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsClientsInvitationsListCall) Do(opts ...googleapi.CallOption) (*ListClientUserInvitationsResponse, error) {
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
	ret := &ListClientUserInvitationsResponse{
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
	//   "description": "Lists all the client users invitations for a client\nwith a given account ID.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.clients.invitations.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Numerical account ID of the client's sponsor buyer. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "Numerical account ID of the client buyer to list invitations for.\n(required)\nYou must either specify a string representation of a\nnumerical account identifier or the `-` character\nto list all the invitations for all the clients\nof a given sponsor buyer.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. Server may return fewer clients than requested.\nIf unspecified, server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListClientUserInvitationsResponse.nextPageToken\nreturned from the previous call to the\nclients.invitations.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/invitations",
	//   "response": {
	//     "$ref": "ListClientUserInvitationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsClientsInvitationsListCall) Pages(ctx context.Context, f func(*ListClientUserInvitationsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.clients.users.get":

type AccountsClientsUsersGetCall struct {
	s               *Service
	accountId       int64
	clientAccountId int64
	userId          int64
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// Get: Retrieves an existing client user.
func (r *AccountsClientsUsersService) Get(accountId int64, clientAccountId int64, userId int64) *AccountsClientsUsersGetCall {
	c := &AccountsClientsUsersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsUsersGetCall) Fields(s ...googleapi.Field) *AccountsClientsUsersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsClientsUsersGetCall) IfNoneMatch(entityTag string) *AccountsClientsUsersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsUsersGetCall) Context(ctx context.Context) *AccountsClientsUsersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsUsersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsUsersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": strconv.FormatInt(c.clientAccountId, 10),
		"userId":          strconv.FormatInt(c.userId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.users.get" call.
// Exactly one of *ClientUser or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ClientUser.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsClientsUsersGetCall) Do(opts ...googleapi.CallOption) (*ClientUser, error) {
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
	ret := &ClientUser{
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
	//   "description": "Retrieves an existing client user.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users/{userId}",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.clients.users.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Numerical account ID of the client's sponsor buyer. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "Numerical account ID of the client buyer\nthat the user to be retrieved is associated with. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Numerical identifier of the user to retrieve. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users/{userId}",
	//   "response": {
	//     "$ref": "ClientUser"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.clients.users.list":

type AccountsClientsUsersListCall struct {
	s               *Service
	accountId       int64
	clientAccountId string
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// List: Lists all the known client users for a specified
// sponsor buyer account ID.
func (r *AccountsClientsUsersService) List(accountId int64, clientAccountId string) *AccountsClientsUsersListCall {
	c := &AccountsClientsUsersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer clients than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsClientsUsersListCall) PageSize(pageSize int64) *AccountsClientsUsersListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListClientUsersResponse.nextPageToken
// returned from the previous call to the
// accounts.clients.users.list method.
func (c *AccountsClientsUsersListCall) PageToken(pageToken string) *AccountsClientsUsersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsUsersListCall) Fields(s ...googleapi.Field) *AccountsClientsUsersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsClientsUsersListCall) IfNoneMatch(entityTag string) *AccountsClientsUsersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsUsersListCall) Context(ctx context.Context) *AccountsClientsUsersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsUsersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsUsersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": c.clientAccountId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.users.list" call.
// Exactly one of *ListClientUsersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListClientUsersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsClientsUsersListCall) Do(opts ...googleapi.CallOption) (*ListClientUsersResponse, error) {
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
	ret := &ListClientUsersResponse{
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
	//   "description": "Lists all the known client users for a specified\nsponsor buyer account ID.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.clients.users.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Numerical account ID of the sponsor buyer of the client to list users for.\n(required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "The account ID of the client buyer to list users for. (required)\nYou must specify either a string representation of a\nnumerical account identifier or the `-` character\nto list all the client users for all the clients\nof a given sponsor buyer.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer clients than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListClientUsersResponse.nextPageToken\nreturned from the previous call to the\naccounts.clients.users.list method.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users",
	//   "response": {
	//     "$ref": "ListClientUsersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsClientsUsersListCall) Pages(ctx context.Context, f func(*ListClientUsersResponse) error) error {
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

// method id "adexchangebuyer2.accounts.clients.users.update":

type AccountsClientsUsersUpdateCall struct {
	s               *Service
	accountId       int64
	clientAccountId int64
	userId          int64
	clientuser      *ClientUser
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Update: Updates an existing client user.
// Only the user status can be changed on update.
func (r *AccountsClientsUsersService) Update(accountId int64, clientAccountId int64, userId int64, clientuser *ClientUser) *AccountsClientsUsersUpdateCall {
	c := &AccountsClientsUsersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.clientAccountId = clientAccountId
	c.userId = userId
	c.clientuser = clientuser
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsClientsUsersUpdateCall) Fields(s ...googleapi.Field) *AccountsClientsUsersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsClientsUsersUpdateCall) Context(ctx context.Context) *AccountsClientsUsersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsClientsUsersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsClientsUsersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.clientuser)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":       strconv.FormatInt(c.accountId, 10),
		"clientAccountId": strconv.FormatInt(c.clientAccountId, 10),
		"userId":          strconv.FormatInt(c.userId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.clients.users.update" call.
// Exactly one of *ClientUser or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ClientUser.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsClientsUsersUpdateCall) Do(opts ...googleapi.CallOption) (*ClientUser, error) {
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
	ret := &ClientUser{
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
	//   "description": "Updates an existing client user.\nOnly the user status can be changed on update.",
	//   "flatPath": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users/{userId}",
	//   "httpMethod": "PUT",
	//   "id": "adexchangebuyer2.accounts.clients.users.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "clientAccountId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Numerical account ID of the client's sponsor buyer. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clientAccountId": {
	//       "description": "Numerical account ID of the client buyer that the user to be retrieved\nis associated with. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Numerical identifier of the user to retrieve. (required)",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/clients/{clientAccountId}/users/{userId}",
	//   "request": {
	//     "$ref": "ClientUser"
	//   },
	//   "response": {
	//     "$ref": "ClientUser"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.creatives.create":

type AccountsCreativesCreateCall struct {
	s          *Service
	accountId  string
	creative   *Creative
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a creative.
func (r *AccountsCreativesService) Create(accountId string, creative *Creative) *AccountsCreativesCreateCall {
	c := &AccountsCreativesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creative = creative
	return c
}

// DuplicateIdMode sets the optional parameter "duplicateIdMode":
// Indicates if multiple creatives can share an ID or not. Default
// is
// NO_DUPLICATES (one ID per creative).
//
// Possible values:
//   "NO_DUPLICATES"
//   "FORCE_ENABLE_DUPLICATE_IDS"
func (c *AccountsCreativesCreateCall) DuplicateIdMode(duplicateIdMode string) *AccountsCreativesCreateCall {
	c.urlParams_.Set("duplicateIdMode", duplicateIdMode)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesCreateCall) Fields(s ...googleapi.Field) *AccountsCreativesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesCreateCall) Context(ctx context.Context) *AccountsCreativesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.creative)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId": c.accountId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.create" call.
// Exactly one of *Creative or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Creative.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsCreativesCreateCall) Do(opts ...googleapi.CallOption) (*Creative, error) {
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
	ret := &Creative{
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
	//   "description": "Creates a creative.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.creatives.create",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account that this creative belongs to.\nCan be used to filter the response of the\ncreatives.list\nmethod.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "duplicateIdMode": {
	//       "description": "Indicates if multiple creatives can share an ID or not. Default is\nNO_DUPLICATES (one ID per creative).",
	//       "enum": [
	//         "NO_DUPLICATES",
	//         "FORCE_ENABLE_DUPLICATE_IDS"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives",
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

// method id "adexchangebuyer2.accounts.creatives.get":

type AccountsCreativesGetCall struct {
	s            *Service
	accountId    string
	creativeId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a creative.
func (r *AccountsCreativesService) Get(accountId string, creativeId string) *AccountsCreativesGetCall {
	c := &AccountsCreativesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creativeId = creativeId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesGetCall) Fields(s ...googleapi.Field) *AccountsCreativesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsCreativesGetCall) IfNoneMatch(entityTag string) *AccountsCreativesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesGetCall) Context(ctx context.Context) *AccountsCreativesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives/{creativeId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":  c.accountId,
		"creativeId": c.creativeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.get" call.
// Exactly one of *Creative or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Creative.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsCreativesGetCall) Do(opts ...googleapi.CallOption) (*Creative, error) {
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
	ret := &Creative{
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
	//   "description": "Gets a creative.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives/{creativeId}",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.creatives.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "creativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account the creative belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeId": {
	//       "description": "The ID of the creative to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives/{creativeId}",
	//   "response": {
	//     "$ref": "Creative"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.creatives.list":

type AccountsCreativesListCall struct {
	s            *Service
	accountId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists creatives.
func (r *AccountsCreativesService) List(accountId string) *AccountsCreativesListCall {
	c := &AccountsCreativesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer creatives than requested
// (due to timeout constraint) even if more are available via another
// call.
// If unspecified, server will pick an appropriate default.
// Acceptable values are 1 to 1000, inclusive.
func (c *AccountsCreativesListCall) PageSize(pageSize int64) *AccountsCreativesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListCreativesResponse.next_page_token
// returned from the previous call to 'ListCreatives' method.
func (c *AccountsCreativesListCall) PageToken(pageToken string) *AccountsCreativesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Query sets the optional parameter "query": An optional query string
// to filter creatives. If no filter is specified,
// all active creatives will be returned.
// Supported queries
// are:
// <ul>
// <li>accountId=<i>account_id_string</i>
// <li>creativeId=<i>cre
// ative_id_string</i>
// <li>dealsStatus: {approved, conditionally_approved, disapproved,
//                    not_checked}
// <li>openAuctionStatus: {approved, conditionally_approved,
// disapproved,
//                           not_checked}
// <li>attribute: {a numeric attribute from the list of
// attributes}
// <li>disapprovalReason: {a reason
// from
// DisapprovalReason
// </ul>
// Example: 'accountId=12345 AND (dealsStatus:disapproved
// AND
// disapprovalReason:unacceptable_content) OR attribute:47'
func (c *AccountsCreativesListCall) Query(query string) *AccountsCreativesListCall {
	c.urlParams_.Set("query", query)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesListCall) Fields(s ...googleapi.Field) *AccountsCreativesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsCreativesListCall) IfNoneMatch(entityTag string) *AccountsCreativesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesListCall) Context(ctx context.Context) *AccountsCreativesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId": c.accountId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.list" call.
// Exactly one of *ListCreativesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListCreativesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsCreativesListCall) Do(opts ...googleapi.CallOption) (*ListCreativesResponse, error) {
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
	ret := &ListCreativesResponse{
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
	//   "description": "Lists creatives.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.creatives.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account to list the creatives from.\nSpecify \"-\" to list all creatives the current user has access to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer creatives than requested\n(due to timeout constraint) even if more are available via another call.\nIf unspecified, server will pick an appropriate default.\nAcceptable values are 1 to 1000, inclusive.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListCreativesResponse.next_page_token\nreturned from the previous call to 'ListCreatives' method.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "query": {
	//       "description": "An optional query string to filter creatives. If no filter is specified,\nall active creatives will be returned.\nSupported queries are:\n\u003cul\u003e\n\u003cli\u003eaccountId=\u003ci\u003eaccount_id_string\u003c/i\u003e\n\u003cli\u003ecreativeId=\u003ci\u003ecreative_id_string\u003c/i\u003e\n\u003cli\u003edealsStatus: {approved, conditionally_approved, disapproved,\n                   not_checked}\n\u003cli\u003eopenAuctionStatus: {approved, conditionally_approved, disapproved,\n                          not_checked}\n\u003cli\u003eattribute: {a numeric attribute from the list of attributes}\n\u003cli\u003edisapprovalReason: {a reason from\nDisapprovalReason\n\u003c/ul\u003e\nExample: 'accountId=12345 AND (dealsStatus:disapproved AND\ndisapprovalReason:unacceptable_content) OR attribute:47'",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives",
	//   "response": {
	//     "$ref": "ListCreativesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsCreativesListCall) Pages(ctx context.Context, f func(*ListCreativesResponse) error) error {
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

// method id "adexchangebuyer2.accounts.creatives.stopWatching":

type AccountsCreativesStopWatchingCall struct {
	s                           *Service
	accountId                   string
	creativeId                  string
	stopwatchingcreativerequest *StopWatchingCreativeRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// StopWatching: Stops watching a creative. Will stop push notifications
// being sent to the
// topics when the creative changes status.
func (r *AccountsCreativesService) StopWatching(accountId string, creativeId string, stopwatchingcreativerequest *StopWatchingCreativeRequest) *AccountsCreativesStopWatchingCall {
	c := &AccountsCreativesStopWatchingCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creativeId = creativeId
	c.stopwatchingcreativerequest = stopwatchingcreativerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesStopWatchingCall) Fields(s ...googleapi.Field) *AccountsCreativesStopWatchingCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesStopWatchingCall) Context(ctx context.Context) *AccountsCreativesStopWatchingCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesStopWatchingCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesStopWatchingCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.stopwatchingcreativerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives/{creativeId}:stopWatching")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":  c.accountId,
		"creativeId": c.creativeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.stopWatching" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsCreativesStopWatchingCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Stops watching a creative. Will stop push notifications being sent to the\ntopics when the creative changes status.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives/{creativeId}:stopWatching",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.creatives.stopWatching",
	//   "parameterOrder": [
	//     "accountId",
	//     "creativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account of the creative to stop notifications for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeId": {
	//       "description": "The creative ID of the creative to stop notifications for.\nSpecify \"-\" to specify stopping account level notifications.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives/{creativeId}:stopWatching",
	//   "request": {
	//     "$ref": "StopWatchingCreativeRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.creatives.update":

type AccountsCreativesUpdateCall struct {
	s          *Service
	accountId  string
	creativeId string
	creative   *Creative
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a creative.
func (r *AccountsCreativesService) Update(accountId string, creativeId string, creative *Creative) *AccountsCreativesUpdateCall {
	c := &AccountsCreativesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creativeId = creativeId
	c.creative = creative
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesUpdateCall) Fields(s ...googleapi.Field) *AccountsCreativesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesUpdateCall) Context(ctx context.Context) *AccountsCreativesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.creative)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives/{creativeId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":  c.accountId,
		"creativeId": c.creativeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.update" call.
// Exactly one of *Creative or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Creative.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsCreativesUpdateCall) Do(opts ...googleapi.CallOption) (*Creative, error) {
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
	ret := &Creative{
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
	//   "description": "Updates a creative.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives/{creativeId}",
	//   "httpMethod": "PUT",
	//   "id": "adexchangebuyer2.accounts.creatives.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "creativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account that this creative belongs to.\nCan be used to filter the response of the\ncreatives.list\nmethod.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeId": {
	//       "description": "The buyer-defined creative ID of this creative.\nCan be used to filter the response of the\ncreatives.list\nmethod.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives/{creativeId}",
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

// method id "adexchangebuyer2.accounts.creatives.watch":

type AccountsCreativesWatchCall struct {
	s                    *Service
	accountId            string
	creativeId           string
	watchcreativerequest *WatchCreativeRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Watch: Watches a creative. Will result in push notifications being
// sent to the
// topic when the creative changes status.
func (r *AccountsCreativesService) Watch(accountId string, creativeId string, watchcreativerequest *WatchCreativeRequest) *AccountsCreativesWatchCall {
	c := &AccountsCreativesWatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creativeId = creativeId
	c.watchcreativerequest = watchcreativerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesWatchCall) Fields(s ...googleapi.Field) *AccountsCreativesWatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesWatchCall) Context(ctx context.Context) *AccountsCreativesWatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesWatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesWatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.watchcreativerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives/{creativeId}:watch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":  c.accountId,
		"creativeId": c.creativeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.watch" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsCreativesWatchCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Watches a creative. Will result in push notifications being sent to the\ntopic when the creative changes status.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives/{creativeId}:watch",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.creatives.watch",
	//   "parameterOrder": [
	//     "accountId",
	//     "creativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account of the creative to watch.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeId": {
	//       "description": "The creative ID to watch for status changes.\nSpecify \"-\" to watch all creatives under the above account.\nIf both creative-level and account-level notifications are\nsent, only a single notification will be sent to the\ncreative-level notification topic.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives/{creativeId}:watch",
	//   "request": {
	//     "$ref": "WatchCreativeRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.creatives.dealAssociations.add":

type AccountsCreativesDealAssociationsAddCall struct {
	s                         *Service
	accountId                 string
	creativeId                string
	adddealassociationrequest *AddDealAssociationRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// Add: Associate an existing deal with a creative.
func (r *AccountsCreativesDealAssociationsService) Add(accountId string, creativeId string, adddealassociationrequest *AddDealAssociationRequest) *AccountsCreativesDealAssociationsAddCall {
	c := &AccountsCreativesDealAssociationsAddCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creativeId = creativeId
	c.adddealassociationrequest = adddealassociationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesDealAssociationsAddCall) Fields(s ...googleapi.Field) *AccountsCreativesDealAssociationsAddCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesDealAssociationsAddCall) Context(ctx context.Context) *AccountsCreativesDealAssociationsAddCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesDealAssociationsAddCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesDealAssociationsAddCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.adddealassociationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations:add")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":  c.accountId,
		"creativeId": c.creativeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.dealAssociations.add" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsCreativesDealAssociationsAddCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Associate an existing deal with a creative.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations:add",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.creatives.dealAssociations.add",
	//   "parameterOrder": [
	//     "accountId",
	//     "creativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account the creative belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeId": {
	//       "description": "The ID of the creative associated with the deal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations:add",
	//   "request": {
	//     "$ref": "AddDealAssociationRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.creatives.dealAssociations.list":

type AccountsCreativesDealAssociationsListCall struct {
	s            *Service
	accountId    string
	creativeId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all creative-deal associations.
func (r *AccountsCreativesDealAssociationsService) List(accountId string, creativeId string) *AccountsCreativesDealAssociationsListCall {
	c := &AccountsCreativesDealAssociationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creativeId = creativeId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// Server may return fewer associations than requested.
// If unspecified, server will pick an appropriate default.
func (c *AccountsCreativesDealAssociationsListCall) PageSize(pageSize int64) *AccountsCreativesDealAssociationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListDealAssociationsResponse.next_page_token
// returned from the previous call to 'ListDealAssociations' method.
func (c *AccountsCreativesDealAssociationsListCall) PageToken(pageToken string) *AccountsCreativesDealAssociationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Query sets the optional parameter "query": An optional query string
// to filter deal associations. If no filter is
// specified, all associations will be returned.
// Supported queries
// are:
// <ul>
// <li>accountId=<i>account_id_string</i>
// <li>creativeId=<i>cre
// ative_id_string</i>
// <li>dealsId=<i>deals_id_string</i>
// <li>dealsStatus
// :{approved, conditionally_approved, disapproved,
//                   not_checked}
// <li>openAuctionStatus:{approved, conditionally_approved,
// disapproved,
//                          not_checked}
// </ul>
// Example: 'dealsId=12345 AND dealsStatus:disapproved'
func (c *AccountsCreativesDealAssociationsListCall) Query(query string) *AccountsCreativesDealAssociationsListCall {
	c.urlParams_.Set("query", query)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesDealAssociationsListCall) Fields(s ...googleapi.Field) *AccountsCreativesDealAssociationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsCreativesDealAssociationsListCall) IfNoneMatch(entityTag string) *AccountsCreativesDealAssociationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesDealAssociationsListCall) Context(ctx context.Context) *AccountsCreativesDealAssociationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesDealAssociationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesDealAssociationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":  c.accountId,
		"creativeId": c.creativeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.dealAssociations.list" call.
// Exactly one of *ListDealAssociationsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListDealAssociationsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsCreativesDealAssociationsListCall) Do(opts ...googleapi.CallOption) (*ListDealAssociationsResponse, error) {
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
	ret := &ListDealAssociationsResponse{
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
	//   "description": "List all creative-deal associations.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.creatives.dealAssociations.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "creativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account to list the associations from.\nSpecify \"-\" to list all creatives the current user has access to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeId": {
	//       "description": "The creative ID to list the associations from.\nSpecify \"-\" to list all creatives under the above account.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. Server may return fewer associations than requested.\nIf unspecified, server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListDealAssociationsResponse.next_page_token\nreturned from the previous call to 'ListDealAssociations' method.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "query": {
	//       "description": "An optional query string to filter deal associations. If no filter is\nspecified, all associations will be returned.\nSupported queries are:\n\u003cul\u003e\n\u003cli\u003eaccountId=\u003ci\u003eaccount_id_string\u003c/i\u003e\n\u003cli\u003ecreativeId=\u003ci\u003ecreative_id_string\u003c/i\u003e\n\u003cli\u003edealsId=\u003ci\u003edeals_id_string\u003c/i\u003e\n\u003cli\u003edealsStatus:{approved, conditionally_approved, disapproved,\n                  not_checked}\n\u003cli\u003eopenAuctionStatus:{approved, conditionally_approved, disapproved,\n                         not_checked}\n\u003c/ul\u003e\nExample: 'dealsId=12345 AND dealsStatus:disapproved'",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations",
	//   "response": {
	//     "$ref": "ListDealAssociationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsCreativesDealAssociationsListCall) Pages(ctx context.Context, f func(*ListDealAssociationsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.creatives.dealAssociations.remove":

type AccountsCreativesDealAssociationsRemoveCall struct {
	s                            *Service
	accountId                    string
	creativeId                   string
	removedealassociationrequest *RemoveDealAssociationRequest
	urlParams_                   gensupport.URLParams
	ctx_                         context.Context
	header_                      http.Header
}

// Remove: Remove the association between a deal and a creative.
func (r *AccountsCreativesDealAssociationsService) Remove(accountId string, creativeId string, removedealassociationrequest *RemoveDealAssociationRequest) *AccountsCreativesDealAssociationsRemoveCall {
	c := &AccountsCreativesDealAssociationsRemoveCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.creativeId = creativeId
	c.removedealassociationrequest = removedealassociationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsCreativesDealAssociationsRemoveCall) Fields(s ...googleapi.Field) *AccountsCreativesDealAssociationsRemoveCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsCreativesDealAssociationsRemoveCall) Context(ctx context.Context) *AccountsCreativesDealAssociationsRemoveCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsCreativesDealAssociationsRemoveCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsCreativesDealAssociationsRemoveCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.removedealassociationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations:remove")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":  c.accountId,
		"creativeId": c.creativeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.creatives.dealAssociations.remove" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsCreativesDealAssociationsRemoveCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Remove the association between a deal and a creative.",
	//   "flatPath": "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations:remove",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.creatives.dealAssociations.remove",
	//   "parameterOrder": [
	//     "accountId",
	//     "creativeId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "The account the creative belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeId": {
	//       "description": "The ID of the creative associated with the deal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/creatives/{creativeId}/dealAssociations:remove",
	//   "request": {
	//     "$ref": "RemoveDealAssociationRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.filterSets.create":

type AccountsFilterSetsCreateCall struct {
	s          *Service
	accountId  int64
	filterset  *FilterSet
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates the specified filter set for the account with the
// given account ID.
func (r *AccountsFilterSetsService) Create(accountId int64, filterset *FilterSet) *AccountsFilterSetsCreateCall {
	c := &AccountsFilterSetsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterset = filterset
	return c
}

// IsTransient sets the optional parameter "isTransient": Whether the
// filter set is transient, or should be persisted indefinitely.
// By default, filter sets are not transient.
// If transient, it will be available for at least 1 hour after
// creation.
func (c *AccountsFilterSetsCreateCall) IsTransient(isTransient bool) *AccountsFilterSetsCreateCall {
	c.urlParams_.Set("isTransient", fmt.Sprint(isTransient))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsCreateCall) Fields(s ...googleapi.Field) *AccountsFilterSetsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsCreateCall) Context(ctx context.Context) *AccountsFilterSetsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.filterset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.create" call.
// Exactly one of *FilterSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *FilterSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsFilterSetsCreateCall) Do(opts ...googleapi.CallOption) (*FilterSet, error) {
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
	ret := &FilterSet{
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
	//   "description": "Creates the specified filter set for the account with the given account ID.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets",
	//   "httpMethod": "POST",
	//   "id": "adexchangebuyer2.accounts.filterSets.create",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "isTransient": {
	//       "description": "Whether the filter set is transient, or should be persisted indefinitely.\nBy default, filter sets are not transient.\nIf transient, it will be available for at least 1 hour after creation.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets",
	//   "request": {
	//     "$ref": "FilterSet"
	//   },
	//   "response": {
	//     "$ref": "FilterSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.filterSets.delete":

type AccountsFilterSetsDeleteCall struct {
	s           *Service
	accountId   int64
	filterSetId int64
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Delete: Deletes the requested filter set from the account with the
// given account
// ID.
func (r *AccountsFilterSetsService) Delete(accountId int64, filterSetId int64) *AccountsFilterSetsDeleteCall {
	c := &AccountsFilterSetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsDeleteCall) Fields(s ...googleapi.Field) *AccountsFilterSetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsDeleteCall) Context(ctx context.Context) *AccountsFilterSetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsFilterSetsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Deletes the requested filter set from the account with the given account\nID.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}",
	//   "httpMethod": "DELETE",
	//   "id": "adexchangebuyer2.accounts.filterSets.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to delete.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.filterSets.get":

type AccountsFilterSetsGetCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the requested filter set for the account with the
// given account
// ID.
func (r *AccountsFilterSetsService) Get(accountId int64, filterSetId int64) *AccountsFilterSetsGetCall {
	c := &AccountsFilterSetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsGetCall) Fields(s ...googleapi.Field) *AccountsFilterSetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsGetCall) IfNoneMatch(entityTag string) *AccountsFilterSetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsGetCall) Context(ctx context.Context) *AccountsFilterSetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.get" call.
// Exactly one of *FilterSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *FilterSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsFilterSetsGetCall) Do(opts ...googleapi.CallOption) (*FilterSet, error) {
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
	ret := &FilterSet{
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
	//   "description": "Retrieves the requested filter set for the account with the given account\nID.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to get.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}",
	//   "response": {
	//     "$ref": "FilterSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// method id "adexchangebuyer2.accounts.filterSets.list":

type AccountsFilterSetsListCall struct {
	s            *Service
	accountId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all filter sets for the account with the given account
// ID.
func (r *AccountsFilterSetsService) List(accountId int64) *AccountsFilterSetsListCall {
	c := &AccountsFilterSetsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsListCall) PageSize(pageSize int64) *AccountsFilterSetsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListFilterSetsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.list
// method.
func (c *AccountsFilterSetsListCall) PageToken(pageToken string) *AccountsFilterSetsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsListCall) Context(ctx context.Context) *AccountsFilterSetsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId": strconv.FormatInt(c.accountId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.list" call.
// Exactly one of *ListFilterSetsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListFilterSetsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsFilterSetsListCall) Do(opts ...googleapi.CallOption) (*ListFilterSetsResponse, error) {
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
	ret := &ListFilterSetsResponse{
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
	//   "description": "Lists all filter sets for the account with the given account ID.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListFilterSetsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets",
	//   "response": {
	//     "$ref": "ListFilterSetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsListCall) Pages(ctx context.Context, f func(*ListFilterSetsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.bidMetrics.list":

type AccountsFilterSetsBidMetricsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all metrics that are measured in terms of number of bids.
func (r *AccountsFilterSetsBidMetricsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsBidMetricsListCall {
	c := &AccountsFilterSetsBidMetricsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsBidMetricsListCall) PageSize(pageSize int64) *AccountsFilterSetsBidMetricsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListBidMetricsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.bidMetrics.list
// method.
func (c *AccountsFilterSetsBidMetricsListCall) PageToken(pageToken string) *AccountsFilterSetsBidMetricsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsBidMetricsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsBidMetricsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsBidMetricsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsBidMetricsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsBidMetricsListCall) Context(ctx context.Context) *AccountsFilterSetsBidMetricsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsBidMetricsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsBidMetricsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidMetrics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.bidMetrics.list" call.
// Exactly one of *ListBidMetricsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListBidMetricsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsFilterSetsBidMetricsListCall) Do(opts ...googleapi.CallOption) (*ListBidMetricsResponse, error) {
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
	ret := &ListBidMetricsResponse{
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
	//   "description": "Lists all metrics that are measured in terms of number of bids.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidMetrics",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.bidMetrics.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListBidMetricsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.bidMetrics.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidMetrics",
	//   "response": {
	//     "$ref": "ListBidMetricsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsBidMetricsListCall) Pages(ctx context.Context, f func(*ListBidMetricsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.bidResponseErrors.list":

type AccountsFilterSetsBidResponseErrorsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all errors that occurred in bid responses, with the number
// of bid
// responses affected for each reason.
func (r *AccountsFilterSetsBidResponseErrorsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsBidResponseErrorsListCall {
	c := &AccountsFilterSetsBidResponseErrorsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsBidResponseErrorsListCall) PageSize(pageSize int64) *AccountsFilterSetsBidResponseErrorsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListBidResponseErrorsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.bidResponseErrors.list
// method.
func (c *AccountsFilterSetsBidResponseErrorsListCall) PageToken(pageToken string) *AccountsFilterSetsBidResponseErrorsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsBidResponseErrorsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsBidResponseErrorsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsBidResponseErrorsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsBidResponseErrorsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsBidResponseErrorsListCall) Context(ctx context.Context) *AccountsFilterSetsBidResponseErrorsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsBidResponseErrorsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsBidResponseErrorsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidResponseErrors")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.bidResponseErrors.list" call.
// Exactly one of *ListBidResponseErrorsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListBidResponseErrorsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsFilterSetsBidResponseErrorsListCall) Do(opts ...googleapi.CallOption) (*ListBidResponseErrorsResponse, error) {
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
	ret := &ListBidResponseErrorsResponse{
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
	//   "description": "List all errors that occurred in bid responses, with the number of bid\nresponses affected for each reason.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidResponseErrors",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.bidResponseErrors.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListBidResponseErrorsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.bidResponseErrors.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidResponseErrors",
	//   "response": {
	//     "$ref": "ListBidResponseErrorsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsBidResponseErrorsListCall) Pages(ctx context.Context, f func(*ListBidResponseErrorsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.bidResponsesWithoutBids.list":

type AccountsFilterSetsBidResponsesWithoutBidsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all reasons for which bid responses were considered to
// have no
// applicable bids, with the number of bid responses affected for each
// reason.
func (r *AccountsFilterSetsBidResponsesWithoutBidsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsBidResponsesWithoutBidsListCall {
	c := &AccountsFilterSetsBidResponsesWithoutBidsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) PageSize(pageSize int64) *AccountsFilterSetsBidResponsesWithoutBidsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListBidResponsesWithoutBidsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.bidResponsesWithoutBids.list
// method.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) PageToken(pageToken string) *AccountsFilterSetsBidResponsesWithoutBidsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsBidResponsesWithoutBidsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsBidResponsesWithoutBidsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) Context(ctx context.Context) *AccountsFilterSetsBidResponsesWithoutBidsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidResponsesWithoutBids")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.bidResponsesWithoutBids.list" call.
// Exactly one of *ListBidResponsesWithoutBidsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListBidResponsesWithoutBidsResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) Do(opts ...googleapi.CallOption) (*ListBidResponsesWithoutBidsResponse, error) {
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
	ret := &ListBidResponsesWithoutBidsResponse{
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
	//   "description": "List all reasons for which bid responses were considered to have no\napplicable bids, with the number of bid responses affected for each reason.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidResponsesWithoutBids",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.bidResponsesWithoutBids.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListBidResponsesWithoutBidsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.bidResponsesWithoutBids.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/bidResponsesWithoutBids",
	//   "response": {
	//     "$ref": "ListBidResponsesWithoutBidsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsBidResponsesWithoutBidsListCall) Pages(ctx context.Context, f func(*ListBidResponsesWithoutBidsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.filteredBidRequests.list":

type AccountsFilterSetsFilteredBidRequestsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all reasons that caused a bid request not to be sent for
// an
// impression, with the number of bid requests not sent for each reason.
func (r *AccountsFilterSetsFilteredBidRequestsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsFilteredBidRequestsListCall {
	c := &AccountsFilterSetsFilteredBidRequestsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) PageSize(pageSize int64) *AccountsFilterSetsFilteredBidRequestsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListFilteredBidRequestsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.filteredBidRequests.list
// method.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) PageToken(pageToken string) *AccountsFilterSetsFilteredBidRequestsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsFilteredBidRequestsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsFilteredBidRequestsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) Context(ctx context.Context) *AccountsFilterSetsFilteredBidRequestsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsFilteredBidRequestsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBidRequests")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.filteredBidRequests.list" call.
// Exactly one of *ListFilteredBidRequestsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListFilteredBidRequestsResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) Do(opts ...googleapi.CallOption) (*ListFilteredBidRequestsResponse, error) {
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
	ret := &ListFilteredBidRequestsResponse{
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
	//   "description": "List all reasons that caused a bid request not to be sent for an\nimpression, with the number of bid requests not sent for each reason.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBidRequests",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.filteredBidRequests.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListFilteredBidRequestsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.filteredBidRequests.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBidRequests",
	//   "response": {
	//     "$ref": "ListFilteredBidRequestsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsFilteredBidRequestsListCall) Pages(ctx context.Context, f func(*ListFilteredBidRequestsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.filteredBids.list":

type AccountsFilterSetsFilteredBidsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all reasons for which bids were filtered, with the number
// of bids
// filtered for each reason.
func (r *AccountsFilterSetsFilteredBidsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsFilteredBidsListCall {
	c := &AccountsFilterSetsFilteredBidsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsFilteredBidsListCall) PageSize(pageSize int64) *AccountsFilterSetsFilteredBidsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListFilteredBidsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.filteredBids.list
// method.
func (c *AccountsFilterSetsFilteredBidsListCall) PageToken(pageToken string) *AccountsFilterSetsFilteredBidsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsFilteredBidsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsFilteredBidsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsFilteredBidsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsFilteredBidsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsFilteredBidsListCall) Context(ctx context.Context) *AccountsFilterSetsFilteredBidsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsFilteredBidsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsFilteredBidsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.filteredBids.list" call.
// Exactly one of *ListFilteredBidsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListFilteredBidsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsFilterSetsFilteredBidsListCall) Do(opts ...googleapi.CallOption) (*ListFilteredBidsResponse, error) {
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
	ret := &ListFilteredBidsResponse{
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
	//   "description": "List all reasons for which bids were filtered, with the number of bids\nfiltered for each reason.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.filteredBids.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListFilteredBidsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.filteredBids.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids",
	//   "response": {
	//     "$ref": "ListFilteredBidsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsFilteredBidsListCall) Pages(ctx context.Context, f func(*ListFilteredBidsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.filteredBids.creatives.list":

type AccountsFilterSetsFilteredBidsCreativesListCall struct {
	s                *Service
	accountId        int64
	filterSetId      int64
	creativeStatusId int64
	urlParams_       gensupport.URLParams
	ifNoneMatch_     string
	ctx_             context.Context
	header_          http.Header
}

// List: List all creatives associated with a specific reason for which
// bids were
// filtered, with the number of bids filtered for each creative.
func (r *AccountsFilterSetsFilteredBidsCreativesService) List(accountId int64, filterSetId int64, creativeStatusId int64) *AccountsFilterSetsFilteredBidsCreativesListCall {
	c := &AccountsFilterSetsFilteredBidsCreativesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	c.creativeStatusId = creativeStatusId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) PageSize(pageSize int64) *AccountsFilterSetsFilteredBidsCreativesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListCreativeStatusBreakdownByCreativeResponse.nextPageToken
// returne
// d from the previous call to
// the
// accounts.filterSets.filteredBids.creatives.list
// method.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) PageToken(pageToken string) *AccountsFilterSetsFilteredBidsCreativesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsFilteredBidsCreativesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsFilteredBidsCreativesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) Context(ctx context.Context) *AccountsFilterSetsFilteredBidsCreativesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsFilteredBidsCreativesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids/{creativeStatusId}/creatives")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":        strconv.FormatInt(c.accountId, 10),
		"filterSetId":      strconv.FormatInt(c.filterSetId, 10),
		"creativeStatusId": strconv.FormatInt(c.creativeStatusId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.filteredBids.creatives.list" call.
// Exactly one of *ListCreativeStatusBreakdownByCreativeResponse or
// error will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *ListCreativeStatusBreakdownByCreativeResponse.ServerResponse.Header
// or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) Do(opts ...googleapi.CallOption) (*ListCreativeStatusBreakdownByCreativeResponse, error) {
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
	ret := &ListCreativeStatusBreakdownByCreativeResponse{
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
	//   "description": "List all creatives associated with a specific reason for which bids were\nfiltered, with the number of bids filtered for each creative.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids/{creativeStatusId}/creatives",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.filteredBids.creatives.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId",
	//     "creativeStatusId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeStatusId": {
	//       "description": "The ID of the creative status for which to retrieve a breakdown by\ncreative.\nSee\n[creative-status-codes](https://developers.google.com/ad-exchange/rtb/downloads/creative-status-codes).",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListCreativeStatusBreakdownByCreativeResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.filteredBids.creatives.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids/{creativeStatusId}/creatives",
	//   "response": {
	//     "$ref": "ListCreativeStatusBreakdownByCreativeResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsFilteredBidsCreativesListCall) Pages(ctx context.Context, f func(*ListCreativeStatusBreakdownByCreativeResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.filteredBids.details.list":

type AccountsFilterSetsFilteredBidsDetailsListCall struct {
	s                *Service
	accountId        int64
	filterSetId      int64
	creativeStatusId int64
	urlParams_       gensupport.URLParams
	ifNoneMatch_     string
	ctx_             context.Context
	header_          http.Header
}

// List: List all details associated with a specific reason for which
// bids were
// filtered, with the number of bids filtered for each detail.
func (r *AccountsFilterSetsFilteredBidsDetailsService) List(accountId int64, filterSetId int64, creativeStatusId int64) *AccountsFilterSetsFilteredBidsDetailsListCall {
	c := &AccountsFilterSetsFilteredBidsDetailsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	c.creativeStatusId = creativeStatusId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) PageSize(pageSize int64) *AccountsFilterSetsFilteredBidsDetailsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListCreativeStatusBreakdownByDetailResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.filteredBids.details.list
// method.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) PageToken(pageToken string) *AccountsFilterSetsFilteredBidsDetailsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsFilteredBidsDetailsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsFilteredBidsDetailsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) Context(ctx context.Context) *AccountsFilterSetsFilteredBidsDetailsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsFilteredBidsDetailsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids/{creativeStatusId}/details")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":        strconv.FormatInt(c.accountId, 10),
		"filterSetId":      strconv.FormatInt(c.filterSetId, 10),
		"creativeStatusId": strconv.FormatInt(c.creativeStatusId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.filteredBids.details.list" call.
// Exactly one of *ListCreativeStatusBreakdownByDetailResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *ListCreativeStatusBreakdownByDetailResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) Do(opts ...googleapi.CallOption) (*ListCreativeStatusBreakdownByDetailResponse, error) {
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
	ret := &ListCreativeStatusBreakdownByDetailResponse{
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
	//   "description": "List all details associated with a specific reason for which bids were\nfiltered, with the number of bids filtered for each detail.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids/{creativeStatusId}/details",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.filteredBids.details.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId",
	//     "creativeStatusId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "creativeStatusId": {
	//       "description": "The ID of the creative status for which to retrieve a breakdown by detail.\nSee\n[creative-status-codes](https://developers.google.com/ad-exchange/rtb/downloads/creative-status-codes).\nDetails are only available for statuses 10, 14, 15, 17, 18, 19, 86, and 87.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListCreativeStatusBreakdownByDetailResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.filteredBids.details.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/filteredBids/{creativeStatusId}/details",
	//   "response": {
	//     "$ref": "ListCreativeStatusBreakdownByDetailResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsFilteredBidsDetailsListCall) Pages(ctx context.Context, f func(*ListCreativeStatusBreakdownByDetailResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.impressionMetrics.list":

type AccountsFilterSetsImpressionMetricsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all metrics that are measured in terms of number of
// impressions.
func (r *AccountsFilterSetsImpressionMetricsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsImpressionMetricsListCall {
	c := &AccountsFilterSetsImpressionMetricsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsImpressionMetricsListCall) PageSize(pageSize int64) *AccountsFilterSetsImpressionMetricsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListImpressionMetricsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.impressionMetrics.list
// method.
func (c *AccountsFilterSetsImpressionMetricsListCall) PageToken(pageToken string) *AccountsFilterSetsImpressionMetricsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsImpressionMetricsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsImpressionMetricsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsImpressionMetricsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsImpressionMetricsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsImpressionMetricsListCall) Context(ctx context.Context) *AccountsFilterSetsImpressionMetricsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsImpressionMetricsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsImpressionMetricsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/impressionMetrics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.impressionMetrics.list" call.
// Exactly one of *ListImpressionMetricsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListImpressionMetricsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsFilterSetsImpressionMetricsListCall) Do(opts ...googleapi.CallOption) (*ListImpressionMetricsResponse, error) {
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
	ret := &ListImpressionMetricsResponse{
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
	//   "description": "Lists all metrics that are measured in terms of number of impressions.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/impressionMetrics",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.impressionMetrics.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListImpressionMetricsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.impressionMetrics.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/impressionMetrics",
	//   "response": {
	//     "$ref": "ListImpressionMetricsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsImpressionMetricsListCall) Pages(ctx context.Context, f func(*ListImpressionMetricsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.losingBids.list":

type AccountsFilterSetsLosingBidsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all reasons for which bids lost in the auction, with the
// number of
// bids that lost for each reason.
func (r *AccountsFilterSetsLosingBidsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsLosingBidsListCall {
	c := &AccountsFilterSetsLosingBidsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsLosingBidsListCall) PageSize(pageSize int64) *AccountsFilterSetsLosingBidsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListLosingBidsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.losingBids.list
// method.
func (c *AccountsFilterSetsLosingBidsListCall) PageToken(pageToken string) *AccountsFilterSetsLosingBidsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsLosingBidsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsLosingBidsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsLosingBidsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsLosingBidsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsLosingBidsListCall) Context(ctx context.Context) *AccountsFilterSetsLosingBidsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsLosingBidsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsLosingBidsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/losingBids")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.losingBids.list" call.
// Exactly one of *ListLosingBidsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLosingBidsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsFilterSetsLosingBidsListCall) Do(opts ...googleapi.CallOption) (*ListLosingBidsResponse, error) {
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
	ret := &ListLosingBidsResponse{
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
	//   "description": "List all reasons for which bids lost in the auction, with the number of\nbids that lost for each reason.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/losingBids",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.losingBids.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListLosingBidsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.losingBids.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/losingBids",
	//   "response": {
	//     "$ref": "ListLosingBidsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsLosingBidsListCall) Pages(ctx context.Context, f func(*ListLosingBidsResponse) error) error {
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

// method id "adexchangebuyer2.accounts.filterSets.nonBillableWinningBids.list":

type AccountsFilterSetsNonBillableWinningBidsListCall struct {
	s            *Service
	accountId    int64
	filterSetId  int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all reasons for which winning bids were not billable, with
// the number
// of bids not billed for each reason.
func (r *AccountsFilterSetsNonBillableWinningBidsService) List(accountId int64, filterSetId int64) *AccountsFilterSetsNonBillableWinningBidsListCall {
	c := &AccountsFilterSetsNonBillableWinningBidsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.accountId = accountId
	c.filterSetId = filterSetId
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// The server may return fewer results than requested.
// If unspecified, the server will pick an appropriate default.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) PageSize(pageSize int64) *AccountsFilterSetsNonBillableWinningBidsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
// Typically, this is the value
// of
// ListNonBillableWinningBidsResponse.nextPageToken
// returned from the previous call to
// the
// accounts.filterSets.nonBillableWinningBids.list
// method.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) PageToken(pageToken string) *AccountsFilterSetsNonBillableWinningBidsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) Fields(s ...googleapi.Field) *AccountsFilterSetsNonBillableWinningBidsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) IfNoneMatch(entityTag string) *AccountsFilterSetsNonBillableWinningBidsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) Context(ctx context.Context) *AccountsFilterSetsNonBillableWinningBidsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsFilterSetsNonBillableWinningBidsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/nonBillableWinningBids")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"accountId":   strconv.FormatInt(c.accountId, 10),
		"filterSetId": strconv.FormatInt(c.filterSetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "adexchangebuyer2.accounts.filterSets.nonBillableWinningBids.list" call.
// Exactly one of *ListNonBillableWinningBidsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListNonBillableWinningBidsResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) Do(opts ...googleapi.CallOption) (*ListNonBillableWinningBidsResponse, error) {
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
	ret := &ListNonBillableWinningBidsResponse{
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
	//   "description": "List all reasons for which winning bids were not billable, with the number\nof bids not billed for each reason.",
	//   "flatPath": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/nonBillableWinningBids",
	//   "httpMethod": "GET",
	//   "id": "adexchangebuyer2.accounts.filterSets.nonBillableWinningBids.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "filterSetId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the buyer.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filterSetId": {
	//       "description": "The ID of the filter set to apply.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. The server may return fewer results than requested.\nIf unspecified, the server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.\nTypically, this is the value of\nListNonBillableWinningBidsResponse.nextPageToken\nreturned from the previous call to the\naccounts.filterSets.nonBillableWinningBids.list\nmethod.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/accounts/{accountId}/filterSets/{filterSetId}/nonBillableWinningBids",
	//   "response": {
	//     "$ref": "ListNonBillableWinningBidsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adexchange.buyer"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsFilterSetsNonBillableWinningBidsListCall) Pages(ctx context.Context, f func(*ListNonBillableWinningBidsResponse) error) error {
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
