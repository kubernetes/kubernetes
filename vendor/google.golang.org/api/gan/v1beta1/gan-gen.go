// Package gan provides access to the Google Affiliate Network API.
//
// See https://developers.google.com/affiliate-network/
//
// Usage example:
//
//   import "google.golang.org/api/gan/v1beta1"
//   ...
//   ganService, err := gan.New(oauthHttpClient)
package gan // import "google.golang.org/api/gan/v1beta1"

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

const apiId = "gan:v1beta1"
const apiName = "gan"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/gan/v1beta1/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Advertisers = NewAdvertisersService(s)
	s.CcOffers = NewCcOffersService(s)
	s.Events = NewEventsService(s)
	s.Links = NewLinksService(s)
	s.Publishers = NewPublishersService(s)
	s.Reports = NewReportsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Advertisers *AdvertisersService

	CcOffers *CcOffersService

	Events *EventsService

	Links *LinksService

	Publishers *PublishersService

	Reports *ReportsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAdvertisersService(s *Service) *AdvertisersService {
	rs := &AdvertisersService{s: s}
	return rs
}

type AdvertisersService struct {
	s *Service
}

func NewCcOffersService(s *Service) *CcOffersService {
	rs := &CcOffersService{s: s}
	return rs
}

type CcOffersService struct {
	s *Service
}

func NewEventsService(s *Service) *EventsService {
	rs := &EventsService{s: s}
	return rs
}

type EventsService struct {
	s *Service
}

func NewLinksService(s *Service) *LinksService {
	rs := &LinksService{s: s}
	return rs
}

type LinksService struct {
	s *Service
}

func NewPublishersService(s *Service) *PublishersService {
	rs := &PublishersService{s: s}
	return rs
}

type PublishersService struct {
	s *Service
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	return rs
}

type ReportsService struct {
	s *Service
}

// Advertiser: An AdvertiserResource.
type Advertiser struct {
	// AllowPublisherCreatedLinks: True if the advertiser allows publisher
	// created links, otherwise false.
	AllowPublisherCreatedLinks bool `json:"allowPublisherCreatedLinks,omitempty"`

	// Category: Category that this advertiser belongs to. A valid list of
	// categories can be found here:
	// http://www.google.com/support/affiliatenetwork/advertiser/bin/answer.py?hl=en&answer=107581
	Category string `json:"category,omitempty"`

	// CommissionDuration: The longest possible length of a commission (how
	// long the cookies on the customer's browser last before they expire).
	CommissionDuration int64 `json:"commissionDuration,omitempty"`

	// ContactEmail: Email that this advertiser would like publishers to
	// contact them with.
	ContactEmail string `json:"contactEmail,omitempty"`

	// ContactPhone: Phone that this advertiser would like publishers to
	// contact them with.
	ContactPhone string `json:"contactPhone,omitempty"`

	// DefaultLinkId: The default link id for this advertiser.
	DefaultLinkId int64 `json:"defaultLinkId,omitempty,string"`

	// Description: Description of the website the advertiser advertises
	// from.
	Description string `json:"description,omitempty"`

	// EpcNinetyDayAverage: The sum of fees paid to publishers divided by
	// the total number of clicks over the past three months. This value
	// should be multiplied by 100 at the time of display.
	EpcNinetyDayAverage *Money `json:"epcNinetyDayAverage,omitempty"`

	// EpcSevenDayAverage: The sum of fees paid to publishers divided by the
	// total number of clicks over the past seven days. This value should be
	// multiplied by 100 at the time of display.
	EpcSevenDayAverage *Money `json:"epcSevenDayAverage,omitempty"`

	// Id: The ID of this advertiser.
	Id int64 `json:"id,omitempty,string"`

	// Item: The requested advertiser.
	Item *Advertiser `json:"item,omitempty"`

	// JoinDate: Date that this advertiser was approved as a Google
	// Affiliate Network advertiser.
	JoinDate string `json:"joinDate,omitempty"`

	// Kind: The kind for an advertiser.
	Kind string `json:"kind,omitempty"`

	// LogoUrl: URL to the logo this advertiser uses on the Google Affiliate
	// Network.
	LogoUrl string `json:"logoUrl,omitempty"`

	// MerchantCenterIds: List of merchant center ids for this advertiser
	MerchantCenterIds googleapi.Int64s `json:"merchantCenterIds,omitempty"`

	// Name: The name of this advertiser.
	Name string `json:"name,omitempty"`

	// PayoutRank: A rank based on commissions paid to publishers over the
	// past 90 days. A number between 1 and 4 where 4 means the top quartile
	// (most money paid) and 1 means the bottom quartile (least money paid).
	PayoutRank string `json:"payoutRank,omitempty"`

	// ProductFeedsEnabled: Allows advertisers to submit product listings to
	// Google Product Search.
	ProductFeedsEnabled bool `json:"productFeedsEnabled,omitempty"`

	// RedirectDomains: List of redirect URLs for this advertiser
	RedirectDomains []string `json:"redirectDomains,omitempty"`

	// SiteUrl: URL of the website this advertiser advertises from.
	SiteUrl string `json:"siteUrl,omitempty"`

	// Status: The status of the requesting publisher's relationship this
	// advertiser.
	Status string `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "AllowPublisherCreatedLinks") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Advertiser) MarshalJSON() ([]byte, error) {
	type noMethod Advertiser
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Advertisers struct {
	// Items: The advertiser list.
	Items []*Advertiser `json:"items,omitempty"`

	// Kind: The kind for a page of advertisers.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The 'pageToken' to pass to the next request to get the
	// next page, if there are more to retrieve.
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

func (s *Advertisers) MarshalJSON() ([]byte, error) {
	type noMethod Advertisers
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CcOffer: A credit card offer. There are many possible result fields.
// We provide two different views of the data, or "projections." The
// "full" projection includes every result field. And the "summary"
// projection, which is the default, includes a smaller subset of the
// fields. The fields included in the summary projection are marked as
// such in their descriptions.
type CcOffer struct {
	// AdditionalCardBenefits: More marketing copy about the card's
	// benefits. A summary field.
	AdditionalCardBenefits []string `json:"additionalCardBenefits,omitempty"`

	// AdditionalCardHolderFee: Any extra fees levied on card holders.
	AdditionalCardHolderFee string `json:"additionalCardHolderFee,omitempty"`

	// AgeMinimum: The youngest a recipient of this card may be.
	AgeMinimum float64 `json:"ageMinimum,omitempty"`

	// AgeMinimumDetails: Text describing the details of the age minimum
	// restriction.
	AgeMinimumDetails string `json:"ageMinimumDetails,omitempty"`

	// AnnualFee: The ongoing annual fee, in dollars.
	AnnualFee float64 `json:"annualFee,omitempty"`

	// AnnualFeeDisplay: Text describing the annual fee, including any
	// difference for the first year. A summary field.
	AnnualFeeDisplay string `json:"annualFeeDisplay,omitempty"`

	// AnnualRewardMaximum: The largest number of units you may accumulate
	// in a year.
	AnnualRewardMaximum float64 `json:"annualRewardMaximum,omitempty"`

	// ApprovedCategories: Possible categories for this card, eg "Low
	// Interest" or "Good." A summary field.
	ApprovedCategories []string `json:"approvedCategories,omitempty"`

	// AprDisplay: Text describing the purchase APR. A summary field.
	AprDisplay string `json:"aprDisplay,omitempty"`

	// BalanceComputationMethod: Text describing how the balance is
	// computed. A summary field.
	BalanceComputationMethod string `json:"balanceComputationMethod,omitempty"`

	// BalanceTransferTerms: Text describing the terms for balance
	// transfers. A summary field.
	BalanceTransferTerms string `json:"balanceTransferTerms,omitempty"`

	// BonusRewards: For cards with rewards programs, extra circumstances
	// whereby additional rewards may be granted.
	BonusRewards []*CcOfferBonusRewards `json:"bonusRewards,omitempty"`

	// CarRentalInsurance: If you get coverage when you use the card for the
	// given activity, this field describes it.
	CarRentalInsurance string `json:"carRentalInsurance,omitempty"`

	// CardBenefits: A list of what the issuer thinks are the most important
	// benefits of the card. Usually summarizes the rewards program, if
	// there is one. A summary field.
	CardBenefits []string `json:"cardBenefits,omitempty"`

	// CardName: The issuer's name for the card, including any trademark or
	// service mark designators. A summary field.
	CardName string `json:"cardName,omitempty"`

	// CardType: What kind of credit card this is, for example secured or
	// unsecured.
	CardType string `json:"cardType,omitempty"`

	// CashAdvanceTerms: Text describing the terms for cash advances. A
	// summary field.
	CashAdvanceTerms string `json:"cashAdvanceTerms,omitempty"`

	// CreditLimitMax: The high end for credit limits the issuer imposes on
	// recipients of this card.
	CreditLimitMax float64 `json:"creditLimitMax,omitempty"`

	// CreditLimitMin: The low end for credit limits the issuer imposes on
	// recipients of this card.
	CreditLimitMin float64 `json:"creditLimitMin,omitempty"`

	// CreditRatingDisplay: Text describing the credit ratings required for
	// recipients of this card, for example "Excellent/Good." A summary
	// field.
	CreditRatingDisplay string `json:"creditRatingDisplay,omitempty"`

	// DefaultFees: Fees for defaulting on your payments.
	DefaultFees []*CcOfferDefaultFees `json:"defaultFees,omitempty"`

	// Disclaimer: A notice that, if present, is referenced via an asterisk
	// by many of the other summary fields. If this field is present, it
	// will always start with an asterisk ("*"), and must be prominently
	// displayed with the offer. A summary field.
	Disclaimer string `json:"disclaimer,omitempty"`

	// EmergencyInsurance: If you get coverage when you use the card for the
	// given activity, this field describes it.
	EmergencyInsurance string `json:"emergencyInsurance,omitempty"`

	// ExistingCustomerOnly: Whether this card is only available to existing
	// customers of the issuer.
	ExistingCustomerOnly bool `json:"existingCustomerOnly,omitempty"`

	// ExtendedWarranty: If you get coverage when you use the card for the
	// given activity, this field describes it.
	ExtendedWarranty string `json:"extendedWarranty,omitempty"`

	// FirstYearAnnualFee: The annual fee for the first year, if different
	// from the ongoing fee. Optional.
	FirstYearAnnualFee float64 `json:"firstYearAnnualFee,omitempty"`

	// FlightAccidentInsurance: If you get coverage when you use the card
	// for the given activity, this field describes it.
	FlightAccidentInsurance string `json:"flightAccidentInsurance,omitempty"`

	// ForeignCurrencyTransactionFee: Fee for each transaction involving a
	// foreign currency.
	ForeignCurrencyTransactionFee string `json:"foreignCurrencyTransactionFee,omitempty"`

	// FraudLiability: If you get coverage when you use the card for the
	// given activity, this field describes it.
	FraudLiability string `json:"fraudLiability,omitempty"`

	// GracePeriodDisplay: Text describing the grace period before finance
	// charges apply. A summary field.
	GracePeriodDisplay string `json:"gracePeriodDisplay,omitempty"`

	// ImageUrl: The link to the image of the card that is shown on Connect
	// Commerce. A summary field.
	ImageUrl string `json:"imageUrl,omitempty"`

	// InitialSetupAndProcessingFee: Fee for setting up the card.
	InitialSetupAndProcessingFee string `json:"initialSetupAndProcessingFee,omitempty"`

	// IntroBalanceTransferTerms: Text describing the terms for introductory
	// period balance transfers. A summary field.
	IntroBalanceTransferTerms string `json:"introBalanceTransferTerms,omitempty"`

	// IntroCashAdvanceTerms: Text describing the terms for introductory
	// period cash advances. A summary field.
	IntroCashAdvanceTerms string `json:"introCashAdvanceTerms,omitempty"`

	// IntroPurchaseTerms: Text describing the terms for introductory period
	// purchases. A summary field.
	IntroPurchaseTerms string `json:"introPurchaseTerms,omitempty"`

	// Issuer: Name of card issuer. A summary field.
	Issuer string `json:"issuer,omitempty"`

	// IssuerId: The Google Affiliate Network ID of the advertiser making
	// this offer.
	IssuerId string `json:"issuerId,omitempty"`

	// IssuerWebsite: The generic link to the issuer's site.
	IssuerWebsite string `json:"issuerWebsite,omitempty"`

	// Kind: The kind for one credit card offer. A summary field.
	Kind string `json:"kind,omitempty"`

	// LandingPageUrl: The link to the issuer's page for this card. A
	// summary field.
	LandingPageUrl string `json:"landingPageUrl,omitempty"`

	// LatePaymentFee: Text describing how much a late payment will cost, eg
	// "up to $35." A summary field.
	LatePaymentFee string `json:"latePaymentFee,omitempty"`

	// LuggageInsurance: If you get coverage when you use the card for the
	// given activity, this field describes it.
	LuggageInsurance string `json:"luggageInsurance,omitempty"`

	// MaxPurchaseRate: The highest interest rate the issuer charges on this
	// card. Expressed as an absolute number, not as a percentage.
	MaxPurchaseRate float64 `json:"maxPurchaseRate,omitempty"`

	// MinPurchaseRate: The lowest interest rate the issuer charges on this
	// card. Expressed as an absolute number, not as a percentage.
	MinPurchaseRate float64 `json:"minPurchaseRate,omitempty"`

	// MinimumFinanceCharge: Text describing how much missing the grace
	// period will cost.
	MinimumFinanceCharge string `json:"minimumFinanceCharge,omitempty"`

	// Network: Which network (eg Visa) the card belongs to. A summary
	// field.
	Network string `json:"network,omitempty"`

	// OfferId: This offer's ID. A summary field.
	OfferId string `json:"offerId,omitempty"`

	// OffersImmediateCashReward: Whether a cash reward program lets you get
	// cash back sooner than end of year or other longish period.
	OffersImmediateCashReward bool `json:"offersImmediateCashReward,omitempty"`

	// OverLimitFee: Fee for exceeding the card's charge limit.
	OverLimitFee string `json:"overLimitFee,omitempty"`

	// ProhibitedCategories: Categories in which the issuer does not wish
	// the card to be displayed. A summary field.
	ProhibitedCategories []string `json:"prohibitedCategories,omitempty"`

	// PurchaseRateAdditionalDetails: Text describing any additional details
	// for the purchase rate. A summary field.
	PurchaseRateAdditionalDetails string `json:"purchaseRateAdditionalDetails,omitempty"`

	// PurchaseRateType: Fixed or variable.
	PurchaseRateType string `json:"purchaseRateType,omitempty"`

	// ReturnedPaymentFee: Text describing the fee for a payment that
	// doesn't clear. A summary field.
	ReturnedPaymentFee string `json:"returnedPaymentFee,omitempty"`

	// RewardPartner: The company that redeems the rewards, if different
	// from the issuer.
	RewardPartner string `json:"rewardPartner,omitempty"`

	// RewardUnit: For cards with rewards programs, the unit of reward. For
	// example, miles, cash back, points.
	RewardUnit string `json:"rewardUnit,omitempty"`

	// Rewards: For cards with rewards programs, detailed rules about how
	// the program works.
	Rewards []*CcOfferRewards `json:"rewards,omitempty"`

	// RewardsExpire: Whether accumulated rewards ever expire.
	RewardsExpire bool `json:"rewardsExpire,omitempty"`

	// RewardsHaveBlackoutDates: For airline miles rewards, tells whether
	// blackout dates apply to the miles.
	RewardsHaveBlackoutDates bool `json:"rewardsHaveBlackoutDates,omitempty"`

	// StatementCopyFee: Fee for requesting a copy of your statement.
	StatementCopyFee string `json:"statementCopyFee,omitempty"`

	// TrackingUrl: The link to ping to register a click on this offer. A
	// summary field.
	TrackingUrl string `json:"trackingUrl,omitempty"`

	// TravelInsurance: If you get coverage when you use the card for the
	// given activity, this field describes it.
	TravelInsurance string `json:"travelInsurance,omitempty"`

	// VariableRatesLastUpdated: When variable rates were last updated.
	VariableRatesLastUpdated string `json:"variableRatesLastUpdated,omitempty"`

	// VariableRatesUpdateFrequency: How often variable rates are updated.
	VariableRatesUpdateFrequency string `json:"variableRatesUpdateFrequency,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AdditionalCardBenefits") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CcOffer) MarshalJSON() ([]byte, error) {
	type noMethod CcOffer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CcOfferBonusRewards struct {
	// Amount: How many units of reward will be granted.
	Amount float64 `json:"amount,omitempty"`

	// Details: The circumstances under which this rule applies, for
	// example, booking a flight via Orbitz.
	Details string `json:"details,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Amount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CcOfferBonusRewards) MarshalJSON() ([]byte, error) {
	type noMethod CcOfferBonusRewards
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CcOfferDefaultFees struct {
	// Category: The type of charge, for example Purchases.
	Category string `json:"category,omitempty"`

	// MaxRate: The highest rate the issuer may charge for defaulting on
	// debt in this category. Expressed as an absolute number, not as a
	// percentage.
	MaxRate float64 `json:"maxRate,omitempty"`

	// MinRate: The lowest rate the issuer may charge for defaulting on debt
	// in this category. Expressed as an absolute number, not as a
	// percentage.
	MinRate float64 `json:"minRate,omitempty"`

	// RateType: Fixed or variable.
	RateType string `json:"rateType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Category") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CcOfferDefaultFees) MarshalJSON() ([]byte, error) {
	type noMethod CcOfferDefaultFees
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CcOfferRewards struct {
	// AdditionalDetails: Other limits, for example, if this rule only
	// applies during an introductory period.
	AdditionalDetails string `json:"additionalDetails,omitempty"`

	// Amount: The number of units rewarded per purchase dollar.
	Amount float64 `json:"amount,omitempty"`

	// Category: The kind of purchases covered by this rule.
	Category string `json:"category,omitempty"`

	// ExpirationMonths: How long rewards granted by this rule last.
	ExpirationMonths float64 `json:"expirationMonths,omitempty"`

	// MaxRewardTier: The maximum purchase amount in the given category for
	// this rule to apply.
	MaxRewardTier float64 `json:"maxRewardTier,omitempty"`

	// MinRewardTier: The minimum purchase amount in the given category
	// before this rule applies.
	MinRewardTier float64 `json:"minRewardTier,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdditionalDetails")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CcOfferRewards) MarshalJSON() ([]byte, error) {
	type noMethod CcOfferRewards
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CcOffers struct {
	// Items: The credit card offers.
	Items []*CcOffer `json:"items,omitempty"`

	// Kind: The kind for a page of credit card offers.
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

func (s *CcOffers) MarshalJSON() ([]byte, error) {
	type noMethod CcOffers
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Event: An EventResource.
type Event struct {
	// AdvertiserId: The ID of advertiser for this event.
	AdvertiserId int64 `json:"advertiserId,omitempty,string"`

	// AdvertiserName: The name of the advertiser for this event.
	AdvertiserName string `json:"advertiserName,omitempty"`

	// ChargeId: The charge ID for this event. Only returned for charge
	// events.
	ChargeId string `json:"chargeId,omitempty"`

	// ChargeType: Charge type of the event
	// (other|slotting_fee|monthly_minimum|tier_bonus|debit|credit). Only
	// returned for charge events.
	ChargeType string `json:"chargeType,omitempty"`

	// CommissionableSales: Amount of money exchanged during the
	// transaction. Only returned for charge and conversion events.
	CommissionableSales *Money `json:"commissionableSales,omitempty"`

	// Earnings: Earnings by the publisher.
	Earnings *Money `json:"earnings,omitempty"`

	// EventDate: The date-time this event was initiated as a RFC 3339
	// date-time value.
	EventDate string `json:"eventDate,omitempty"`

	// Kind: The kind for one event.
	Kind string `json:"kind,omitempty"`

	// MemberId: The ID of the member attached to this event. Only returned
	// for conversion events.
	MemberId string `json:"memberId,omitempty"`

	// ModifyDate: The date-time this event was last modified as a RFC 3339
	// date-time value.
	ModifyDate string `json:"modifyDate,omitempty"`

	// NetworkFee: Fee that the advertiser paid to the Google Affiliate
	// Network.
	NetworkFee *Money `json:"networkFee,omitempty"`

	// OrderId: The order ID for this event. Only returned for conversion
	// events.
	OrderId string `json:"orderId,omitempty"`

	// Products: Products associated with the event.
	Products []*EventProducts `json:"products,omitempty"`

	// PublisherFee: Fee that the advertiser paid to the publisher.
	PublisherFee *Money `json:"publisherFee,omitempty"`

	// PublisherId: The ID of the publisher for this event.
	PublisherId int64 `json:"publisherId,omitempty,string"`

	// PublisherName: The name of the publisher for this event.
	PublisherName string `json:"publisherName,omitempty"`

	// Status: Status of the event (active|canceled). Only returned for
	// charge and conversion events.
	Status string `json:"status,omitempty"`

	// Type: Type of the event (action|transaction|charge).
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdvertiserId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Event) MarshalJSON() ([]byte, error) {
	type noMethod Event
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type EventProducts struct {
	// CategoryId: Id of the category this product belongs to.
	CategoryId string `json:"categoryId,omitempty"`

	// CategoryName: Name of the category this product belongs to.
	CategoryName string `json:"categoryName,omitempty"`

	// Earnings: Amount earned by the publisher on this product.
	Earnings *Money `json:"earnings,omitempty"`

	// NetworkFee: Fee that the advertiser paid to the Google Affiliate
	// Network for this product.
	NetworkFee *Money `json:"networkFee,omitempty"`

	// PublisherFee: Fee that the advertiser paid to the publisehr for this
	// product.
	PublisherFee *Money `json:"publisherFee,omitempty"`

	// Quantity: Quantity of this product bought/exchanged.
	Quantity int64 `json:"quantity,omitempty,string"`

	// Sku: Sku of this product.
	Sku string `json:"sku,omitempty"`

	// SkuName: Sku name of this product.
	SkuName string `json:"skuName,omitempty"`

	// UnitPrice: Price per unit of this product.
	UnitPrice *Money `json:"unitPrice,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CategoryId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EventProducts) MarshalJSON() ([]byte, error) {
	type noMethod EventProducts
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Events struct {
	// Items: The event list.
	Items []*Event `json:"items,omitempty"`

	// Kind: The kind for a page of events.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The 'pageToken' to pass to the next request to get the
	// next page, if there are more to retrieve.
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

func (s *Events) MarshalJSON() ([]byte, error) {
	type noMethod Events
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Link: A LinkResource.
type Link struct {
	// AdvertiserId: The advertiser id for the advertiser who owns this
	// link.
	AdvertiserId int64 `json:"advertiserId,omitempty,string"`

	// Authorship: Authorship
	Authorship string `json:"authorship,omitempty"`

	// Availability: Availability.
	Availability string `json:"availability,omitempty"`

	// ClickTrackingUrl: Tracking url for clicks.
	ClickTrackingUrl string `json:"clickTrackingUrl,omitempty"`

	// CreateDate: Date that this link was created.
	CreateDate string `json:"createDate,omitempty"`

	// Description: Description.
	Description string `json:"description,omitempty"`

	// DestinationUrl: The destination URL for the link.
	DestinationUrl string `json:"destinationUrl,omitempty"`

	// Duration: Duration
	Duration string `json:"duration,omitempty"`

	// EndDate: Date that this link becomes inactive.
	EndDate string `json:"endDate,omitempty"`

	// EpcNinetyDayAverage: The sum of fees paid to publishers divided by
	// the total number of clicks over the past three months on this link.
	// This value should be multiplied by 100 at the time of display.
	EpcNinetyDayAverage *Money `json:"epcNinetyDayAverage,omitempty"`

	// EpcSevenDayAverage: The sum of fees paid to publishers divided by the
	// total number of clicks over the past seven days on this link. This
	// value should be multiplied by 100 at the time of display.
	EpcSevenDayAverage *Money `json:"epcSevenDayAverage,omitempty"`

	// Id: The ID of this link.
	Id int64 `json:"id,omitempty,string"`

	// ImageAltText: image alt text.
	ImageAltText string `json:"imageAltText,omitempty"`

	// ImpressionTrackingUrl: Tracking url for impressions.
	ImpressionTrackingUrl string `json:"impressionTrackingUrl,omitempty"`

	// IsActive: Flag for if this link is active.
	IsActive bool `json:"isActive,omitempty"`

	// Kind: The kind for one entity.
	Kind string `json:"kind,omitempty"`

	// LinkType: The link type.
	LinkType string `json:"linkType,omitempty"`

	// Name: The logical name for this link.
	Name string `json:"name,omitempty"`

	// PromotionType: Promotion Type
	PromotionType string `json:"promotionType,omitempty"`

	// SpecialOffers: Special offers on the link.
	SpecialOffers *LinkSpecialOffers `json:"specialOffers,omitempty"`

	// StartDate: Date that this link becomes active.
	StartDate string `json:"startDate,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AdvertiserId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Link) MarshalJSON() ([]byte, error) {
	type noMethod Link
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LinkSpecialOffers: Special offers on the link.
type LinkSpecialOffers struct {
	// FreeGift: Whether there is a free gift
	FreeGift bool `json:"freeGift,omitempty"`

	// FreeShipping: Whether there is free shipping
	FreeShipping bool `json:"freeShipping,omitempty"`

	// FreeShippingMin: Minimum purchase amount for free shipping promotion
	FreeShippingMin *Money `json:"freeShippingMin,omitempty"`

	// PercentOff: Percent off on the purchase
	PercentOff float64 `json:"percentOff,omitempty"`

	// PercentOffMin: Minimum purchase amount for percent off promotion
	PercentOffMin *Money `json:"percentOffMin,omitempty"`

	// PriceCut: Price cut on the purchase
	PriceCut *Money `json:"priceCut,omitempty"`

	// PriceCutMin: Minimum purchase amount for price cut promotion
	PriceCutMin *Money `json:"priceCutMin,omitempty"`

	// PromotionCodes: List of promotion code associated with the link
	PromotionCodes []string `json:"promotionCodes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FreeGift") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LinkSpecialOffers) MarshalJSON() ([]byte, error) {
	type noMethod LinkSpecialOffers
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Links struct {
	// Items: The links.
	Items []*Link `json:"items,omitempty"`

	// Kind: The kind for a page of links.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The next page token.
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

func (s *Links) MarshalJSON() ([]byte, error) {
	type noMethod Links
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Money: An ApiMoneyProto.
type Money struct {
	// Amount: The amount of money.
	Amount float64 `json:"amount,omitempty"`

	// CurrencyCode: The 3-letter code of the currency in question.
	CurrencyCode string `json:"currencyCode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Amount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Money) MarshalJSON() ([]byte, error) {
	type noMethod Money
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Publisher: A PublisherResource.
type Publisher struct {
	// Classification: Classification that this publisher belongs to. See
	// this link for all publisher classifications:
	// http://www.google.com/support/affiliatenetwork/advertiser/bin/answer.py?hl=en&answer=107625&ctx=cb&src=cb&cbid=-k5fihzthfaik&cbrank=4
	Classification string `json:"classification,omitempty"`

	// EpcNinetyDayAverage: The sum of fees paid to this publisher divided
	// by the total number of clicks over the past three months. Values are
	// multiplied by 100 for display purposes.
	EpcNinetyDayAverage *Money `json:"epcNinetyDayAverage,omitempty"`

	// EpcSevenDayAverage: The sum of fees paid to this publisher divided by
	// the total number of clicks over the past seven days. Values are
	// multiplied by 100 for display purposes.
	EpcSevenDayAverage *Money `json:"epcSevenDayAverage,omitempty"`

	// Id: The ID of this publisher.
	Id int64 `json:"id,omitempty,string"`

	// Item: The requested publisher.
	Item *Publisher `json:"item,omitempty"`

	// JoinDate: Date that this publisher was approved as a Google Affiliate
	// Network publisher.
	JoinDate string `json:"joinDate,omitempty"`

	// Kind: The kind for a publisher.
	Kind string `json:"kind,omitempty"`

	// Name: The name of this publisher.
	Name string `json:"name,omitempty"`

	// PayoutRank: A rank based on commissions paid to this publisher over
	// the past 90 days. A number between 1 and 4 where 4 means the top
	// quartile (most money paid) and 1 means the bottom quartile (least
	// money paid).
	PayoutRank string `json:"payoutRank,omitempty"`

	// Sites: Websites that this publisher uses to advertise.
	Sites []string `json:"sites,omitempty"`

	// Status: The status of the requesting advertiser's relationship with
	// this publisher.
	Status string `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Classification") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Publisher) MarshalJSON() ([]byte, error) {
	type noMethod Publisher
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Publishers struct {
	// Items: The entity list.
	Items []*Publisher `json:"items,omitempty"`

	// Kind: The kind for a page of entities.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The 'pageToken' to pass to the next request to get the
	// next page, if there are more to retrieve.
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

func (s *Publishers) MarshalJSON() ([]byte, error) {
	type noMethod Publishers
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Report: A ReportResource representing a report of a certain type
// either for an advertiser or publisher.
type Report struct {
	// ColumnNames: The column names for the report
	ColumnNames []string `json:"column_names,omitempty"`

	// EndDate: The end of the date range for this report, exclusive.
	EndDate string `json:"end_date,omitempty"`

	// Kind: The kind for a report.
	Kind string `json:"kind,omitempty"`

	// MatchingRowCount: The number of matching rows before paging is
	// applied.
	MatchingRowCount int64 `json:"matching_row_count,omitempty,string"`

	// Rows: The rows of data for the report
	Rows [][]interface{} `json:"rows,omitempty"`

	// StartDate: The start of the date range for this report, inclusive.
	StartDate string `json:"start_date,omitempty"`

	// TotalsRows: The totals rows for the report
	TotalsRows [][]interface{} `json:"totals_rows,omitempty"`

	// Type: The report type.
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ColumnNames") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Report) MarshalJSON() ([]byte, error) {
	type noMethod Report
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "gan.advertisers.get":

type AdvertisersGetCall struct {
	s            *Service
	role         string
	roleId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves data about a single advertiser if that the requesting
// advertiser/publisher has access to it. Only publishers can lookup
// advertisers. Advertisers can request information about themselves by
// omitting the advertiserId query parameter.
func (r *AdvertisersService) Get(role string, roleId string) *AdvertisersGetCall {
	c := &AdvertisersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	return c
}

// AdvertiserId sets the optional parameter "advertiserId": The ID of
// the advertiser to look up.
func (c *AdvertisersGetCall) AdvertiserId(advertiserId string) *AdvertisersGetCall {
	c.urlParams_.Set("advertiserId", advertiserId)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AdvertisersGetCall) QuotaUser(quotaUser string) *AdvertisersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AdvertisersGetCall) UserIP(userIP string) *AdvertisersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AdvertisersGetCall) Fields(s ...googleapi.Field) *AdvertisersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AdvertisersGetCall) IfNoneMatch(entityTag string) *AdvertisersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AdvertisersGetCall) Context(ctx context.Context) *AdvertisersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AdvertisersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/advertiser")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
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

// Do executes the "gan.advertisers.get" call.
// Exactly one of *Advertiser or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Advertiser.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AdvertisersGetCall) Do() (*Advertiser, error) {
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
	ret := &Advertiser{
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
	//   "description": "Retrieves data about a single advertiser if that the requesting advertiser/publisher has access to it. Only publishers can lookup advertisers. Advertisers can request information about themselves by omitting the advertiserId query parameter.",
	//   "httpMethod": "GET",
	//   "id": "gan.advertisers.get",
	//   "parameterOrder": [
	//     "role",
	//     "roleId"
	//   ],
	//   "parameters": {
	//     "advertiserId": {
	//       "description": "The ID of the advertiser to look up. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/advertiser",
	//   "response": {
	//     "$ref": "Advertiser"
	//   }
	// }

}

// method id "gan.advertisers.list":

type AdvertisersListCall struct {
	s            *Service
	role         string
	roleId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves data about all advertisers that the requesting
// advertiser/publisher has access to.
func (r *AdvertisersService) List(role string, roleId string) *AdvertisersListCall {
	c := &AdvertisersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	return c
}

// AdvertiserCategory sets the optional parameter "advertiserCategory":
// Caret(^) delimted list of advertiser categories. Valid categories are
// defined here:
// http://www.google.com/support/affiliatenetwork/advertiser/bin/answer.py?hl=en&answer=107581. Filters out all advertisers not in one of the given advertiser
// categories.
func (c *AdvertisersListCall) AdvertiserCategory(advertiserCategory string) *AdvertisersListCall {
	c.urlParams_.Set("advertiserCategory", advertiserCategory)
	return c
}

// MaxResults sets the optional parameter "maxResults": Max number of
// items to return in this page.  Defaults to 20.
func (c *AdvertisersListCall) MaxResults(maxResults int64) *AdvertisersListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// MinNinetyDayEpc sets the optional parameter "minNinetyDayEpc":
// Filters out all advertisers that have a ninety day EPC average lower
// than the given value (inclusive). Min value: 0.0.
func (c *AdvertisersListCall) MinNinetyDayEpc(minNinetyDayEpc float64) *AdvertisersListCall {
	c.urlParams_.Set("minNinetyDayEpc", fmt.Sprint(minNinetyDayEpc))
	return c
}

// MinPayoutRank sets the optional parameter "minPayoutRank": A value
// between 1 and 4, where 1 represents the quartile of advertisers with
// the lowest ranks and 4 represents the quartile of advertisers with
// the highest ranks. Filters out all advertisers with a lower rank than
// the given quartile. For example if a 2 was given only advertisers
// with a payout rank of 25 or higher would be included.
func (c *AdvertisersListCall) MinPayoutRank(minPayoutRank int64) *AdvertisersListCall {
	c.urlParams_.Set("minPayoutRank", fmt.Sprint(minPayoutRank))
	return c
}

// MinSevenDayEpc sets the optional parameter "minSevenDayEpc": Filters
// out all advertisers that have a seven day EPC average lower than the
// given value (inclusive). Min value: 0.0.
func (c *AdvertisersListCall) MinSevenDayEpc(minSevenDayEpc float64) *AdvertisersListCall {
	c.urlParams_.Set("minSevenDayEpc", fmt.Sprint(minSevenDayEpc))
	return c
}

// PageToken sets the optional parameter "pageToken": The value of
// 'nextPageToken' from the previous page.
func (c *AdvertisersListCall) PageToken(pageToken string) *AdvertisersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AdvertisersListCall) QuotaUser(quotaUser string) *AdvertisersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// RelationshipStatus sets the optional parameter "relationshipStatus":
// Filters out all advertisers for which do not have the given
// relationship status with the requesting publisher.
//
// Possible values:
//   "approved" - An advertiser that has approved your application.
//   "available" - An advertiser program that's accepting new
// publishers.
//   "deactivated" - Deactivated means either the advertiser has removed
// you from their program, or it could also mean that you chose to
// remove yourself from the advertiser's program.
//   "declined" - An advertiser that did not approve your application.
//   "pending" - An advertiser program that you've already applied to,
// but they haven't yet decided to approve or decline your application.
func (c *AdvertisersListCall) RelationshipStatus(relationshipStatus string) *AdvertisersListCall {
	c.urlParams_.Set("relationshipStatus", relationshipStatus)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AdvertisersListCall) UserIP(userIP string) *AdvertisersListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AdvertisersListCall) Fields(s ...googleapi.Field) *AdvertisersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AdvertisersListCall) IfNoneMatch(entityTag string) *AdvertisersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AdvertisersListCall) Context(ctx context.Context) *AdvertisersListCall {
	c.ctx_ = ctx
	return c
}

func (c *AdvertisersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/advertisers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
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

// Do executes the "gan.advertisers.list" call.
// Exactly one of *Advertisers or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Advertisers.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AdvertisersListCall) Do() (*Advertisers, error) {
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
	ret := &Advertisers{
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
	//   "description": "Retrieves data about all advertisers that the requesting advertiser/publisher has access to.",
	//   "httpMethod": "GET",
	//   "id": "gan.advertisers.list",
	//   "parameterOrder": [
	//     "role",
	//     "roleId"
	//   ],
	//   "parameters": {
	//     "advertiserCategory": {
	//       "description": "Caret(^) delimted list of advertiser categories. Valid categories are defined here: http://www.google.com/support/affiliatenetwork/advertiser/bin/answer.py?hl=en\u0026answer=107581. Filters out all advertisers not in one of the given advertiser categories. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Max number of items to return in this page. Optional. Defaults to 20.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "minNinetyDayEpc": {
	//       "description": "Filters out all advertisers that have a ninety day EPC average lower than the given value (inclusive). Min value: 0.0. Optional.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "minPayoutRank": {
	//       "description": "A value between 1 and 4, where 1 represents the quartile of advertisers with the lowest ranks and 4 represents the quartile of advertisers with the highest ranks. Filters out all advertisers with a lower rank than the given quartile. For example if a 2 was given only advertisers with a payout rank of 25 or higher would be included. Optional.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "4",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "minSevenDayEpc": {
	//       "description": "Filters out all advertisers that have a seven day EPC average lower than the given value (inclusive). Min value: 0.0. Optional.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "pageToken": {
	//       "description": "The value of 'nextPageToken' from the previous page. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "relationshipStatus": {
	//       "description": "Filters out all advertisers for which do not have the given relationship status with the requesting publisher.",
	//       "enum": [
	//         "approved",
	//         "available",
	//         "deactivated",
	//         "declined",
	//         "pending"
	//       ],
	//       "enumDescriptions": [
	//         "An advertiser that has approved your application.",
	//         "An advertiser program that's accepting new publishers.",
	//         "Deactivated means either the advertiser has removed you from their program, or it could also mean that you chose to remove yourself from the advertiser's program.",
	//         "An advertiser that did not approve your application.",
	//         "An advertiser program that you've already applied to, but they haven't yet decided to approve or decline your application."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/advertisers",
	//   "response": {
	//     "$ref": "Advertisers"
	//   }
	// }

}

// method id "gan.ccOffers.list":

type CcOffersListCall struct {
	s            *Service
	publisher    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves credit card offers for the given publisher.
func (r *CcOffersService) List(publisher string) *CcOffersListCall {
	c := &CcOffersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.publisher = publisher
	return c
}

// Advertiser sets the optional parameter "advertiser": The advertiser
// ID of a card issuer whose offers to include. Optional, may be
// repeated.
func (c *CcOffersListCall) Advertiser(advertiser ...string) *CcOffersListCall {
	c.urlParams_.SetMulti("advertiser", append([]string{}, advertiser...))
	return c
}

// Projection sets the optional parameter "projection": The set of
// fields to return.
//
// Possible values:
//   "full" - Include all offer fields. This is the default.
//   "summary" - Include only the basic fields needed to display an
// offer.
func (c *CcOffersListCall) Projection(projection string) *CcOffersListCall {
	c.urlParams_.Set("projection", projection)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CcOffersListCall) QuotaUser(quotaUser string) *CcOffersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CcOffersListCall) UserIP(userIP string) *CcOffersListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CcOffersListCall) Fields(s ...googleapi.Field) *CcOffersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CcOffersListCall) IfNoneMatch(entityTag string) *CcOffersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CcOffersListCall) Context(ctx context.Context) *CcOffersListCall {
	c.ctx_ = ctx
	return c
}

func (c *CcOffersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "publishers/{publisher}/ccOffers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"publisher": c.publisher,
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

// Do executes the "gan.ccOffers.list" call.
// Exactly one of *CcOffers or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *CcOffers.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CcOffersListCall) Do() (*CcOffers, error) {
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
	ret := &CcOffers{
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
	//   "description": "Retrieves credit card offers for the given publisher.",
	//   "httpMethod": "GET",
	//   "id": "gan.ccOffers.list",
	//   "parameterOrder": [
	//     "publisher"
	//   ],
	//   "parameters": {
	//     "advertiser": {
	//       "description": "The advertiser ID of a card issuer whose offers to include. Optional, may be repeated.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "The set of fields to return.",
	//       "enum": [
	//         "full",
	//         "summary"
	//       ],
	//       "enumDescriptions": [
	//         "Include all offer fields. This is the default.",
	//         "Include only the basic fields needed to display an offer."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "publisher": {
	//       "description": "The ID of the publisher in question.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "publishers/{publisher}/ccOffers",
	//   "response": {
	//     "$ref": "CcOffers"
	//   }
	// }

}

// method id "gan.events.list":

type EventsListCall struct {
	s            *Service
	role         string
	roleId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves event data for a given advertiser/publisher.
func (r *EventsService) List(role string, roleId string) *EventsListCall {
	c := &EventsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	return c
}

// AdvertiserId sets the optional parameter "advertiserId": Caret(^)
// delimited list of advertiser IDs. Filters out all events that do not
// reference one of the given advertiser IDs. Only used when under
// publishers role.
func (c *EventsListCall) AdvertiserId(advertiserId string) *EventsListCall {
	c.urlParams_.Set("advertiserId", advertiserId)
	return c
}

// ChargeType sets the optional parameter "chargeType": Filters out all
// charge events that are not of the given charge type. Valid values:
// 'other', 'slotting_fee', 'monthly_minimum', 'tier_bonus', 'credit',
// 'debit'.
//
// Possible values:
//   "credit" - A credit increases the publisher's payout amount and
// decreases the advertiser's invoice amount.
//   "debit" - A debit reduces the publisher's payout and increases the
// advertiser's invoice amount.
//   "monthly_minimum" - A payment made to Google by an advertiser as a
// minimum monthly network fee.
//   "other" - Catch all. Default if unset
//   "slotting_fee" - A one time payment made from an advertiser to a
// publisher.
//   "tier_bonus" - A payment from an advertiser to a publisher for the
// publisher maintaining a high tier level
func (c *EventsListCall) ChargeType(chargeType string) *EventsListCall {
	c.urlParams_.Set("chargeType", chargeType)
	return c
}

// EventDateMax sets the optional parameter "eventDateMax": Filters out
// all events later than given date.  Defaults to 24 hours after
// eventMin.
func (c *EventsListCall) EventDateMax(eventDateMax string) *EventsListCall {
	c.urlParams_.Set("eventDateMax", eventDateMax)
	return c
}

// EventDateMin sets the optional parameter "eventDateMin": Filters out
// all events earlier than given date.  Defaults to 24 hours from
// current date/time.
func (c *EventsListCall) EventDateMin(eventDateMin string) *EventsListCall {
	c.urlParams_.Set("eventDateMin", eventDateMin)
	return c
}

// LinkId sets the optional parameter "linkId": Caret(^) delimited list
// of link IDs. Filters out all events that do not reference one of the
// given link IDs.
func (c *EventsListCall) LinkId(linkId string) *EventsListCall {
	c.urlParams_.Set("linkId", linkId)
	return c
}

// MaxResults sets the optional parameter "maxResults": Max number of
// offers to return in this page.  Defaults to 20.
func (c *EventsListCall) MaxResults(maxResults int64) *EventsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// MemberId sets the optional parameter "memberId": Caret(^) delimited
// list of member IDs. Filters out all events that do not reference one
// of the given member IDs.
func (c *EventsListCall) MemberId(memberId string) *EventsListCall {
	c.urlParams_.Set("memberId", memberId)
	return c
}

// ModifyDateMax sets the optional parameter "modifyDateMax": Filters
// out all events modified later than given date.  Defaults to 24 hours
// after modifyDateMin, if modifyDateMin is explicitly set.
func (c *EventsListCall) ModifyDateMax(modifyDateMax string) *EventsListCall {
	c.urlParams_.Set("modifyDateMax", modifyDateMax)
	return c
}

// ModifyDateMin sets the optional parameter "modifyDateMin": Filters
// out all events modified earlier than given date.  Defaults to 24
// hours before the current modifyDateMax, if modifyDateMax is
// explicitly set.
func (c *EventsListCall) ModifyDateMin(modifyDateMin string) *EventsListCall {
	c.urlParams_.Set("modifyDateMin", modifyDateMin)
	return c
}

// OrderId sets the optional parameter "orderId": Caret(^) delimited
// list of order IDs. Filters out all events that do not reference one
// of the given order IDs.
func (c *EventsListCall) OrderId(orderId string) *EventsListCall {
	c.urlParams_.Set("orderId", orderId)
	return c
}

// PageToken sets the optional parameter "pageToken": The value of
// 'nextPageToken' from the previous page.
func (c *EventsListCall) PageToken(pageToken string) *EventsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ProductCategory sets the optional parameter "productCategory":
// Caret(^) delimited list of product categories. Filters out all events
// that do not reference a product in one of the given product
// categories.
func (c *EventsListCall) ProductCategory(productCategory string) *EventsListCall {
	c.urlParams_.Set("productCategory", productCategory)
	return c
}

// PublisherId sets the optional parameter "publisherId": Caret(^)
// delimited list of publisher IDs. Filters out all events that do not
// reference one of the given publishers IDs. Only used when under
// advertiser role.
func (c *EventsListCall) PublisherId(publisherId string) *EventsListCall {
	c.urlParams_.Set("publisherId", publisherId)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EventsListCall) QuotaUser(quotaUser string) *EventsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Sku sets the optional parameter "sku": Caret(^) delimited list of
// SKUs. Filters out all events that do not reference one of the given
// SKU.
func (c *EventsListCall) Sku(sku string) *EventsListCall {
	c.urlParams_.Set("sku", sku)
	return c
}

// Status sets the optional parameter "status": Filters out all events
// that do not have the given status. Valid values: 'active',
// 'canceled'.
//
// Possible values:
//   "active" - Event is currently active.
//   "canceled" - Event is currently canceled.
func (c *EventsListCall) Status(status string) *EventsListCall {
	c.urlParams_.Set("status", status)
	return c
}

// Type sets the optional parameter "type": Filters out all events that
// are not of the given type. Valid values: 'action', 'transaction',
// 'charge'.
//
// Possible values:
//   "action" - The completion of an application, sign-up, or other
// process. For example, an action occurs if a user clicks an ad for a
// credit card and completes an application for that card.
//   "charge" - A charge event is typically a payment between an
// advertiser, publisher or Google.
//   "transaction" - A conversion event, typically an e-commerce
// transaction. Some advertisers use a transaction to record other types
// of events, such as magazine subscriptions.
func (c *EventsListCall) Type(type_ string) *EventsListCall {
	c.urlParams_.Set("type", type_)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EventsListCall) UserIP(userIP string) *EventsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EventsListCall) Fields(s ...googleapi.Field) *EventsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EventsListCall) IfNoneMatch(entityTag string) *EventsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EventsListCall) Context(ctx context.Context) *EventsListCall {
	c.ctx_ = ctx
	return c
}

func (c *EventsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/events")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
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

// Do executes the "gan.events.list" call.
// Exactly one of *Events or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Events.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EventsListCall) Do() (*Events, error) {
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
	ret := &Events{
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
	//   "description": "Retrieves event data for a given advertiser/publisher.",
	//   "httpMethod": "GET",
	//   "id": "gan.events.list",
	//   "parameterOrder": [
	//     "role",
	//     "roleId"
	//   ],
	//   "parameters": {
	//     "advertiserId": {
	//       "description": "Caret(^) delimited list of advertiser IDs. Filters out all events that do not reference one of the given advertiser IDs. Only used when under publishers role. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "chargeType": {
	//       "description": "Filters out all charge events that are not of the given charge type. Valid values: 'other', 'slotting_fee', 'monthly_minimum', 'tier_bonus', 'credit', 'debit'. Optional.",
	//       "enum": [
	//         "credit",
	//         "debit",
	//         "monthly_minimum",
	//         "other",
	//         "slotting_fee",
	//         "tier_bonus"
	//       ],
	//       "enumDescriptions": [
	//         "A credit increases the publisher's payout amount and decreases the advertiser's invoice amount.",
	//         "A debit reduces the publisher's payout and increases the advertiser's invoice amount.",
	//         "A payment made to Google by an advertiser as a minimum monthly network fee.",
	//         "Catch all. Default if unset",
	//         "A one time payment made from an advertiser to a publisher.",
	//         "A payment from an advertiser to a publisher for the publisher maintaining a high tier level"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "eventDateMax": {
	//       "description": "Filters out all events later than given date. Optional. Defaults to 24 hours after eventMin.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "eventDateMin": {
	//       "description": "Filters out all events earlier than given date. Optional. Defaults to 24 hours from current date/time.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Caret(^) delimited list of link IDs. Filters out all events that do not reference one of the given link IDs. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Max number of offers to return in this page. Optional. Defaults to 20.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "memberId": {
	//       "description": "Caret(^) delimited list of member IDs. Filters out all events that do not reference one of the given member IDs. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "modifyDateMax": {
	//       "description": "Filters out all events modified later than given date. Optional. Defaults to 24 hours after modifyDateMin, if modifyDateMin is explicitly set.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "modifyDateMin": {
	//       "description": "Filters out all events modified earlier than given date. Optional. Defaults to 24 hours before the current modifyDateMax, if modifyDateMax is explicitly set.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "orderId": {
	//       "description": "Caret(^) delimited list of order IDs. Filters out all events that do not reference one of the given order IDs. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The value of 'nextPageToken' from the previous page. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "productCategory": {
	//       "description": "Caret(^) delimited list of product categories. Filters out all events that do not reference a product in one of the given product categories. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "publisherId": {
	//       "description": "Caret(^) delimited list of publisher IDs. Filters out all events that do not reference one of the given publishers IDs. Only used when under advertiser role. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sku": {
	//       "description": "Caret(^) delimited list of SKUs. Filters out all events that do not reference one of the given SKU. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "status": {
	//       "description": "Filters out all events that do not have the given status. Valid values: 'active', 'canceled'. Optional.",
	//       "enum": [
	//         "active",
	//         "canceled"
	//       ],
	//       "enumDescriptions": [
	//         "Event is currently active.",
	//         "Event is currently canceled."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "type": {
	//       "description": "Filters out all events that are not of the given type. Valid values: 'action', 'transaction', 'charge'. Optional.",
	//       "enum": [
	//         "action",
	//         "charge",
	//         "transaction"
	//       ],
	//       "enumDescriptions": [
	//         "The completion of an application, sign-up, or other process. For example, an action occurs if a user clicks an ad for a credit card and completes an application for that card.",
	//         "A charge event is typically a payment between an advertiser, publisher or Google.",
	//         "A conversion event, typically an e-commerce transaction. Some advertisers use a transaction to record other types of events, such as magazine subscriptions."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/events",
	//   "response": {
	//     "$ref": "Events"
	//   }
	// }

}

// method id "gan.links.get":

type LinksGetCall struct {
	s            *Service
	role         string
	roleId       string
	linkId       int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves data about a single link if the requesting
// advertiser/publisher has access to it. Advertisers can look up their
// own links. Publishers can look up visible links or links belonging to
// advertisers they are in a relationship with.
func (r *LinksService) Get(role string, roleId string, linkId int64) *LinksGetCall {
	c := &LinksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	c.linkId = linkId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *LinksGetCall) QuotaUser(quotaUser string) *LinksGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *LinksGetCall) UserIP(userIP string) *LinksGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LinksGetCall) Fields(s ...googleapi.Field) *LinksGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *LinksGetCall) IfNoneMatch(entityTag string) *LinksGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LinksGetCall) Context(ctx context.Context) *LinksGetCall {
	c.ctx_ = ctx
	return c
}

func (c *LinksGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/link/{linkId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
		"linkId": strconv.FormatInt(c.linkId, 10),
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

// Do executes the "gan.links.get" call.
// Exactly one of *Link or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Link.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *LinksGetCall) Do() (*Link, error) {
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
	ret := &Link{
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
	//   "description": "Retrieves data about a single link if the requesting advertiser/publisher has access to it. Advertisers can look up their own links. Publishers can look up visible links or links belonging to advertisers they are in a relationship with.",
	//   "httpMethod": "GET",
	//   "id": "gan.links.get",
	//   "parameterOrder": [
	//     "role",
	//     "roleId",
	//     "linkId"
	//   ],
	//   "parameters": {
	//     "linkId": {
	//       "description": "The ID of the link to look up.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/link/{linkId}",
	//   "response": {
	//     "$ref": "Link"
	//   }
	// }

}

// method id "gan.links.insert":

type LinksInsertCall struct {
	s          *Service
	role       string
	roleId     string
	link       *Link
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Inserts a new link.
func (r *LinksService) Insert(role string, roleId string, link *Link) *LinksInsertCall {
	c := &LinksInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	c.link = link
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *LinksInsertCall) QuotaUser(quotaUser string) *LinksInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *LinksInsertCall) UserIP(userIP string) *LinksInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LinksInsertCall) Fields(s ...googleapi.Field) *LinksInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LinksInsertCall) Context(ctx context.Context) *LinksInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *LinksInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.link)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/link")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gan.links.insert" call.
// Exactly one of *Link or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Link.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *LinksInsertCall) Do() (*Link, error) {
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
	ret := &Link{
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
	//   "description": "Inserts a new link.",
	//   "httpMethod": "POST",
	//   "id": "gan.links.insert",
	//   "parameterOrder": [
	//     "role",
	//     "roleId"
	//   ],
	//   "parameters": {
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/link",
	//   "request": {
	//     "$ref": "Link"
	//   },
	//   "response": {
	//     "$ref": "Link"
	//   }
	// }

}

// method id "gan.links.list":

type LinksListCall struct {
	s            *Service
	role         string
	roleId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves all links that match the query parameters.
func (r *LinksService) List(role string, roleId string) *LinksListCall {
	c := &LinksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	return c
}

// AdvertiserId sets the optional parameter "advertiserId": Limits the
// resulting links to the ones belonging to the listed advertisers.
func (c *LinksListCall) AdvertiserId(advertiserId ...int64) *LinksListCall {
	var advertiserId_ []string
	for _, v := range advertiserId {
		advertiserId_ = append(advertiserId_, fmt.Sprint(v))
	}
	c.urlParams_.SetMulti("advertiserId", advertiserId_)
	return c
}

// AssetSize sets the optional parameter "assetSize": The size of the
// given asset.
func (c *LinksListCall) AssetSize(assetSize ...string) *LinksListCall {
	c.urlParams_.SetMulti("assetSize", append([]string{}, assetSize...))
	return c
}

// Authorship sets the optional parameter "authorship": The role of the
// author of the link.
//
// Possible values:
//   "advertiser"
//   "publisher"
func (c *LinksListCall) Authorship(authorship string) *LinksListCall {
	c.urlParams_.Set("authorship", authorship)
	return c
}

// CreateDateMax sets the optional parameter "createDateMax": The end of
// the create date range.
func (c *LinksListCall) CreateDateMax(createDateMax string) *LinksListCall {
	c.urlParams_.Set("createDateMax", createDateMax)
	return c
}

// CreateDateMin sets the optional parameter "createDateMin": The
// beginning of the create date range.
func (c *LinksListCall) CreateDateMin(createDateMin string) *LinksListCall {
	c.urlParams_.Set("createDateMin", createDateMin)
	return c
}

// LinkType sets the optional parameter "linkType": The type of the
// link.
//
// Possible values:
//   "banner"
//   "text"
func (c *LinksListCall) LinkType(linkType string) *LinksListCall {
	c.urlParams_.Set("linkType", linkType)
	return c
}

// MaxResults sets the optional parameter "maxResults": Max number of
// items to return in this page.  Defaults to 20.
func (c *LinksListCall) MaxResults(maxResults int64) *LinksListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The value of
// 'nextPageToken' from the previous page.
func (c *LinksListCall) PageToken(pageToken string) *LinksListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// PromotionType sets the optional parameter "promotionType": The
// promotion type.
//
// Possible values:
//   "coupon"
//   "free_gift"
//   "free_shipping"
//   "percent_off"
//   "price_cut"
func (c *LinksListCall) PromotionType(promotionType ...string) *LinksListCall {
	c.urlParams_.SetMulti("promotionType", append([]string{}, promotionType...))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *LinksListCall) QuotaUser(quotaUser string) *LinksListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// RelationshipStatus sets the optional parameter "relationshipStatus":
// The status of the relationship.
//
// Possible values:
//   "approved"
//   "available"
func (c *LinksListCall) RelationshipStatus(relationshipStatus string) *LinksListCall {
	c.urlParams_.Set("relationshipStatus", relationshipStatus)
	return c
}

// SearchText sets the optional parameter "searchText": Field for full
// text search across title and merchandising text, supports link id
// search.
func (c *LinksListCall) SearchText(searchText string) *LinksListCall {
	c.urlParams_.Set("searchText", searchText)
	return c
}

// StartDateMax sets the optional parameter "startDateMax": The end of
// the start date range.
func (c *LinksListCall) StartDateMax(startDateMax string) *LinksListCall {
	c.urlParams_.Set("startDateMax", startDateMax)
	return c
}

// StartDateMin sets the optional parameter "startDateMin": The
// beginning of the start date range.
func (c *LinksListCall) StartDateMin(startDateMin string) *LinksListCall {
	c.urlParams_.Set("startDateMin", startDateMin)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *LinksListCall) UserIP(userIP string) *LinksListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LinksListCall) Fields(s ...googleapi.Field) *LinksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *LinksListCall) IfNoneMatch(entityTag string) *LinksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LinksListCall) Context(ctx context.Context) *LinksListCall {
	c.ctx_ = ctx
	return c
}

func (c *LinksListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/links")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
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

// Do executes the "gan.links.list" call.
// Exactly one of *Links or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Links.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *LinksListCall) Do() (*Links, error) {
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
	ret := &Links{
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
	//   "description": "Retrieves all links that match the query parameters.",
	//   "httpMethod": "GET",
	//   "id": "gan.links.list",
	//   "parameterOrder": [
	//     "role",
	//     "roleId"
	//   ],
	//   "parameters": {
	//     "advertiserId": {
	//       "description": "Limits the resulting links to the ones belonging to the listed advertisers.",
	//       "format": "int64",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "assetSize": {
	//       "description": "The size of the given asset.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "authorship": {
	//       "description": "The role of the author of the link.",
	//       "enum": [
	//         "advertiser",
	//         "publisher"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "createDateMax": {
	//       "description": "The end of the create date range.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "createDateMin": {
	//       "description": "The beginning of the create date range.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "linkType": {
	//       "description": "The type of the link.",
	//       "enum": [
	//         "banner",
	//         "text"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Max number of items to return in this page. Optional. Defaults to 20.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of 'nextPageToken' from the previous page. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "promotionType": {
	//       "description": "The promotion type.",
	//       "enum": [
	//         "coupon",
	//         "free_gift",
	//         "free_shipping",
	//         "percent_off",
	//         "price_cut"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "relationshipStatus": {
	//       "description": "The status of the relationship.",
	//       "enum": [
	//         "approved",
	//         "available"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "searchText": {
	//       "description": "Field for full text search across title and merchandising text, supports link id search.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startDateMax": {
	//       "description": "The end of the start date range.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startDateMin": {
	//       "description": "The beginning of the start date range.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/links",
	//   "response": {
	//     "$ref": "Links"
	//   }
	// }

}

// method id "gan.publishers.get":

type PublishersGetCall struct {
	s            *Service
	role         string
	roleId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves data about a single advertiser if that the requesting
// advertiser/publisher has access to it. Only advertisers can look up
// publishers. Publishers can request information about themselves by
// omitting the publisherId query parameter.
func (r *PublishersService) Get(role string, roleId string) *PublishersGetCall {
	c := &PublishersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	return c
}

// PublisherId sets the optional parameter "publisherId": The ID of the
// publisher to look up.
func (c *PublishersGetCall) PublisherId(publisherId string) *PublishersGetCall {
	c.urlParams_.Set("publisherId", publisherId)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PublishersGetCall) QuotaUser(quotaUser string) *PublishersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PublishersGetCall) UserIP(userIP string) *PublishersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PublishersGetCall) Fields(s ...googleapi.Field) *PublishersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PublishersGetCall) IfNoneMatch(entityTag string) *PublishersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PublishersGetCall) Context(ctx context.Context) *PublishersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *PublishersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/publisher")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
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

// Do executes the "gan.publishers.get" call.
// Exactly one of *Publisher or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Publisher.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PublishersGetCall) Do() (*Publisher, error) {
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
	ret := &Publisher{
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
	//   "description": "Retrieves data about a single advertiser if that the requesting advertiser/publisher has access to it. Only advertisers can look up publishers. Publishers can request information about themselves by omitting the publisherId query parameter.",
	//   "httpMethod": "GET",
	//   "id": "gan.publishers.get",
	//   "parameterOrder": [
	//     "role",
	//     "roleId"
	//   ],
	//   "parameters": {
	//     "publisherId": {
	//       "description": "The ID of the publisher to look up. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/publisher",
	//   "response": {
	//     "$ref": "Publisher"
	//   }
	// }

}

// method id "gan.publishers.list":

type PublishersListCall struct {
	s            *Service
	role         string
	roleId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves data about all publishers that the requesting
// advertiser/publisher has access to.
func (r *PublishersService) List(role string, roleId string) *PublishersListCall {
	c := &PublishersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	return c
}

// MaxResults sets the optional parameter "maxResults": Max number of
// items to return in this page.  Defaults to 20.
func (c *PublishersListCall) MaxResults(maxResults int64) *PublishersListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// MinNinetyDayEpc sets the optional parameter "minNinetyDayEpc":
// Filters out all publishers that have a ninety day EPC average lower
// than the given value (inclusive). Min value: 0.0.
func (c *PublishersListCall) MinNinetyDayEpc(minNinetyDayEpc float64) *PublishersListCall {
	c.urlParams_.Set("minNinetyDayEpc", fmt.Sprint(minNinetyDayEpc))
	return c
}

// MinPayoutRank sets the optional parameter "minPayoutRank": A value
// between 1 and 4, where 1 represents the quartile of publishers with
// the lowest ranks and 4 represents the quartile of publishers with the
// highest ranks. Filters out all publishers with a lower rank than the
// given quartile. For example if a 2 was given only publishers with a
// payout rank of 25 or higher would be included.
func (c *PublishersListCall) MinPayoutRank(minPayoutRank int64) *PublishersListCall {
	c.urlParams_.Set("minPayoutRank", fmt.Sprint(minPayoutRank))
	return c
}

// MinSevenDayEpc sets the optional parameter "minSevenDayEpc": Filters
// out all publishers that have a seven day EPC average lower than the
// given value (inclusive). Min value 0.0.
func (c *PublishersListCall) MinSevenDayEpc(minSevenDayEpc float64) *PublishersListCall {
	c.urlParams_.Set("minSevenDayEpc", fmt.Sprint(minSevenDayEpc))
	return c
}

// PageToken sets the optional parameter "pageToken": The value of
// 'nextPageToken' from the previous page.
func (c *PublishersListCall) PageToken(pageToken string) *PublishersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// PublisherCategory sets the optional parameter "publisherCategory":
// Caret(^) delimted list of publisher categories. Valid categories:
// (unclassified|community_and_content|shopping_and_promotion|loyalty_and
// _rewards|network|search_specialist|comparison_shopping|email).
// Filters out all publishers not in one of the given advertiser
// categories.
func (c *PublishersListCall) PublisherCategory(publisherCategory string) *PublishersListCall {
	c.urlParams_.Set("publisherCategory", publisherCategory)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PublishersListCall) QuotaUser(quotaUser string) *PublishersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// RelationshipStatus sets the optional parameter "relationshipStatus":
// Filters out all publishers for which do not have the given
// relationship status with the requesting publisher.
//
// Possible values:
//   "approved" - Publishers you've approved to your program.
//   "available" - Publishers available for you to recruit.
//   "deactivated" - A publisher that you terminated from your program.
// Publishers also have the ability to remove themselves from your
// program.
//   "declined" - A publisher that you did not approve to your program.
//   "pending" - Publishers that have applied to your program. We
// recommend reviewing and deciding on pending publishers on a weekly
// basis.
func (c *PublishersListCall) RelationshipStatus(relationshipStatus string) *PublishersListCall {
	c.urlParams_.Set("relationshipStatus", relationshipStatus)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PublishersListCall) UserIP(userIP string) *PublishersListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PublishersListCall) Fields(s ...googleapi.Field) *PublishersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PublishersListCall) IfNoneMatch(entityTag string) *PublishersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PublishersListCall) Context(ctx context.Context) *PublishersListCall {
	c.ctx_ = ctx
	return c
}

func (c *PublishersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/publishers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":   c.role,
		"roleId": c.roleId,
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

// Do executes the "gan.publishers.list" call.
// Exactly one of *Publishers or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Publishers.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PublishersListCall) Do() (*Publishers, error) {
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
	ret := &Publishers{
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
	//   "description": "Retrieves data about all publishers that the requesting advertiser/publisher has access to.",
	//   "httpMethod": "GET",
	//   "id": "gan.publishers.list",
	//   "parameterOrder": [
	//     "role",
	//     "roleId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Max number of items to return in this page. Optional. Defaults to 20.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "minNinetyDayEpc": {
	//       "description": "Filters out all publishers that have a ninety day EPC average lower than the given value (inclusive). Min value: 0.0. Optional.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "minPayoutRank": {
	//       "description": "A value between 1 and 4, where 1 represents the quartile of publishers with the lowest ranks and 4 represents the quartile of publishers with the highest ranks. Filters out all publishers with a lower rank than the given quartile. For example if a 2 was given only publishers with a payout rank of 25 or higher would be included. Optional.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "4",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "minSevenDayEpc": {
	//       "description": "Filters out all publishers that have a seven day EPC average lower than the given value (inclusive). Min value 0.0. Optional.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "pageToken": {
	//       "description": "The value of 'nextPageToken' from the previous page. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "publisherCategory": {
	//       "description": "Caret(^) delimted list of publisher categories. Valid categories: (unclassified|community_and_content|shopping_and_promotion|loyalty_and_rewards|network|search_specialist|comparison_shopping|email). Filters out all publishers not in one of the given advertiser categories. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "relationshipStatus": {
	//       "description": "Filters out all publishers for which do not have the given relationship status with the requesting publisher.",
	//       "enum": [
	//         "approved",
	//         "available",
	//         "deactivated",
	//         "declined",
	//         "pending"
	//       ],
	//       "enumDescriptions": [
	//         "Publishers you've approved to your program.",
	//         "Publishers available for you to recruit.",
	//         "A publisher that you terminated from your program. Publishers also have the ability to remove themselves from your program.",
	//         "A publisher that you did not approve to your program.",
	//         "Publishers that have applied to your program. We recommend reviewing and deciding on pending publishers on a weekly basis."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/publishers",
	//   "response": {
	//     "$ref": "Publishers"
	//   }
	// }

}

// method id "gan.reports.get":

type ReportsGetCall struct {
	s            *Service
	role         string
	roleId       string
	reportType   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves a report of the specified type.
func (r *ReportsService) Get(role string, roleId string, reportType string) *ReportsGetCall {
	c := &ReportsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.role = role
	c.roleId = roleId
	c.reportType = reportType
	return c
}

// AdvertiserId sets the optional parameter "advertiserId": The IDs of
// the advertisers to look up, if applicable.
func (c *ReportsGetCall) AdvertiserId(advertiserId ...string) *ReportsGetCall {
	c.urlParams_.SetMulti("advertiserId", append([]string{}, advertiserId...))
	return c
}

// CalculateTotals sets the optional parameter "calculateTotals":
// Whether or not to calculate totals rows.
func (c *ReportsGetCall) CalculateTotals(calculateTotals bool) *ReportsGetCall {
	c.urlParams_.Set("calculateTotals", fmt.Sprint(calculateTotals))
	return c
}

// EndDate sets the optional parameter "endDate": The end date
// (exclusive), in RFC 3339 format, for the report data to be returned.
// Defaults to one day after startDate, if that is given, or today.
func (c *ReportsGetCall) EndDate(endDate string) *ReportsGetCall {
	c.urlParams_.Set("endDate", endDate)
	return c
}

// EventType sets the optional parameter "eventType": Filters out all
// events that are not of the given type. Valid values: 'action',
// 'transaction', or 'charge'.
//
// Possible values:
//   "action" - Event type is action.
//   "charge" - Event type is charge.
//   "transaction" - Event type is transaction.
func (c *ReportsGetCall) EventType(eventType string) *ReportsGetCall {
	c.urlParams_.Set("eventType", eventType)
	return c
}

// LinkId sets the optional parameter "linkId": Filters to capture one
// of given link IDs.
func (c *ReportsGetCall) LinkId(linkId ...string) *ReportsGetCall {
	c.urlParams_.SetMulti("linkId", append([]string{}, linkId...))
	return c
}

// MaxResults sets the optional parameter "maxResults": Max number of
// items to return in this page.  Defaults to return all results.
func (c *ReportsGetCall) MaxResults(maxResults int64) *ReportsGetCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// OrderId sets the optional parameter "orderId": Filters to capture one
// of the given order IDs.
func (c *ReportsGetCall) OrderId(orderId ...string) *ReportsGetCall {
	c.urlParams_.SetMulti("orderId", append([]string{}, orderId...))
	return c
}

// PublisherId sets the optional parameter "publisherId": The IDs of the
// publishers to look up, if applicable.
func (c *ReportsGetCall) PublisherId(publisherId ...string) *ReportsGetCall {
	c.urlParams_.SetMulti("publisherId", append([]string{}, publisherId...))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsGetCall) QuotaUser(quotaUser string) *ReportsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// StartDate sets the optional parameter "startDate": The start date
// (inclusive), in RFC 3339 format, for the report data to be returned.
// Defaults to one day before endDate, if that is given, or yesterday.
func (c *ReportsGetCall) StartDate(startDate string) *ReportsGetCall {
	c.urlParams_.Set("startDate", startDate)
	return c
}

// StartIndex sets the optional parameter "startIndex": Offset on which
// to return results when paging.
func (c *ReportsGetCall) StartIndex(startIndex int64) *ReportsGetCall {
	c.urlParams_.Set("startIndex", fmt.Sprint(startIndex))
	return c
}

// Status sets the optional parameter "status": Filters out all events
// that do not have the given status. Valid values: 'active',
// 'canceled', or 'invalid'.
//
// Possible values:
//   "active" - Event is currently active.
//   "canceled" - Event is currently canceled.
//   "invalid" - Event is currently invalid.
func (c *ReportsGetCall) Status(status string) *ReportsGetCall {
	c.urlParams_.Set("status", status)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsGetCall) UserIP(userIP string) *ReportsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsGetCall) Fields(s ...googleapi.Field) *ReportsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReportsGetCall) IfNoneMatch(entityTag string) *ReportsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsGetCall) Context(ctx context.Context) *ReportsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{role}/{roleId}/report/{reportType}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"role":       c.role,
		"roleId":     c.roleId,
		"reportType": c.reportType,
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

// Do executes the "gan.reports.get" call.
// Exactly one of *Report or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Report.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ReportsGetCall) Do() (*Report, error) {
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
	ret := &Report{
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
	//   "description": "Retrieves a report of the specified type.",
	//   "httpMethod": "GET",
	//   "id": "gan.reports.get",
	//   "parameterOrder": [
	//     "role",
	//     "roleId",
	//     "reportType"
	//   ],
	//   "parameters": {
	//     "advertiserId": {
	//       "description": "The IDs of the advertisers to look up, if applicable.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "calculateTotals": {
	//       "description": "Whether or not to calculate totals rows. Optional.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "endDate": {
	//       "description": "The end date (exclusive), in RFC 3339 format, for the report data to be returned. Defaults to one day after startDate, if that is given, or today. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "eventType": {
	//       "description": "Filters out all events that are not of the given type. Valid values: 'action', 'transaction', or 'charge'. Optional.",
	//       "enum": [
	//         "action",
	//         "charge",
	//         "transaction"
	//       ],
	//       "enumDescriptions": [
	//         "Event type is action.",
	//         "Event type is charge.",
	//         "Event type is transaction."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Filters to capture one of given link IDs. Optional.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Max number of items to return in this page. Optional. Defaults to return all results.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "orderId": {
	//       "description": "Filters to capture one of the given order IDs. Optional.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "publisherId": {
	//       "description": "The IDs of the publishers to look up, if applicable.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "reportType": {
	//       "description": "The type of report being requested. Valid values: 'order_delta'. Required.",
	//       "enum": [
	//         "order_delta"
	//       ],
	//       "enumDescriptions": [
	//         "The order delta report type."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "role": {
	//       "description": "The role of the requester. Valid values: 'advertisers' or 'publishers'.",
	//       "enum": [
	//         "advertisers",
	//         "publishers"
	//       ],
	//       "enumDescriptions": [
	//         "The requester is requesting as an advertiser.",
	//         "The requester is requesting as a publisher."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "roleId": {
	//       "description": "The ID of the requesting advertiser or publisher.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "The start date (inclusive), in RFC 3339 format, for the report data to be returned. Defaults to one day before endDate, if that is given, or yesterday. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Offset on which to return results when paging. Optional.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "status": {
	//       "description": "Filters out all events that do not have the given status. Valid values: 'active', 'canceled', or 'invalid'. Optional.",
	//       "enum": [
	//         "active",
	//         "canceled",
	//         "invalid"
	//       ],
	//       "enumDescriptions": [
	//         "Event is currently active.",
	//         "Event is currently canceled.",
	//         "Event is currently invalid."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{role}/{roleId}/report/{reportType}",
	//   "response": {
	//     "$ref": "Report"
	//   }
	// }

}
