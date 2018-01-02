// Package content provides access to the Content API for Shopping.
//
// See https://developers.google.com/shopping-content
//
// Usage example:
//
//   import "google.golang.org/api/content/v2sandbox"
//   ...
//   contentService, err := content.New(oauthHttpClient)
package content // import "google.golang.org/api/content/v2sandbox"

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

const apiId = "content:v2sandbox"
const apiName = "content"
const apiVersion = "v2sandbox"
const basePath = "https://www.googleapis.com/content/v2sandbox/"

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
	s.Orders = NewOrdersService(s)
	return s, nil
}

type APIService struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Orders *OrdersService
}

func (s *APIService) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewOrdersService(s *APIService) *OrdersService {
	rs := &OrdersService{s: s}
	return rs
}

type OrdersService struct {
	s *APIService
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
