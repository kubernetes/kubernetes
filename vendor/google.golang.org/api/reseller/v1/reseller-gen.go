// Package reseller provides access to the Enterprise Apps Reseller API.
//
// See https://developers.google.com/google-apps/reseller/
//
// Usage example:
//
//   import "google.golang.org/api/reseller/v1"
//   ...
//   resellerService, err := reseller.New(oauthHttpClient)
package reseller // import "google.golang.org/api/reseller/v1"

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

const apiId = "reseller:v1"
const apiName = "reseller"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/apps/reseller/v1/"

// OAuth2 scopes used by this API.
const (
	// Manage users on your domain
	AppsOrderScope = "https://www.googleapis.com/auth/apps.order"

	// Manage users on your domain
	AppsOrderReadonlyScope = "https://www.googleapis.com/auth/apps.order.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Customers = NewCustomersService(s)
	s.Subscriptions = NewSubscriptionsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Customers *CustomersService

	Subscriptions *SubscriptionsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewCustomersService(s *Service) *CustomersService {
	rs := &CustomersService{s: s}
	return rs
}

type CustomersService struct {
	s *Service
}

func NewSubscriptionsService(s *Service) *SubscriptionsService {
	rs := &SubscriptionsService{s: s}
	return rs
}

type SubscriptionsService struct {
	s *Service
}

// Address: JSON template for address of a customer.
type Address struct {
	// AddressLine1: Address line 1 of the address.
	AddressLine1 string `json:"addressLine1,omitempty"`

	// AddressLine2: Address line 2 of the address.
	AddressLine2 string `json:"addressLine2,omitempty"`

	// AddressLine3: Address line 3 of the address.
	AddressLine3 string `json:"addressLine3,omitempty"`

	// ContactName: Name of the contact person.
	ContactName string `json:"contactName,omitempty"`

	// CountryCode: ISO 3166 country code.
	CountryCode string `json:"countryCode,omitempty"`

	// Kind: Identifies the resource as a customer address.
	Kind string `json:"kind,omitempty"`

	// Locality: Name of the locality. This is in accordance with -
	// http://portablecontacts.net/draft-spec.html#address_element.
	Locality string `json:"locality,omitempty"`

	// OrganizationName: Name of the organization.
	OrganizationName string `json:"organizationName,omitempty"`

	// PostalCode: The postal code. This is in accordance with -
	// http://portablecontacts.net/draft-spec.html#address_element.
	PostalCode string `json:"postalCode,omitempty"`

	// Region: Name of the region. This is in accordance with -
	// http://portablecontacts.net/draft-spec.html#address_element.
	Region string `json:"region,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AddressLine1") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Address) MarshalJSON() ([]byte, error) {
	type noMethod Address
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ChangePlanRequest: JSON template for the ChangePlan rpc request.
type ChangePlanRequest struct {
	// Kind: Identifies the resource as a subscription change plan request.
	Kind string `json:"kind,omitempty"`

	// PlanName: Name of the plan to change to.
	PlanName string `json:"planName,omitempty"`

	// PurchaseOrderId: Purchase order id for your order tracking purposes.
	PurchaseOrderId string `json:"purchaseOrderId,omitempty"`

	// Seats: Number/Limit of seats in the new plan.
	Seats *Seats `json:"seats,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ChangePlanRequest) MarshalJSON() ([]byte, error) {
	type noMethod ChangePlanRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Customer: JSON template for a customer.
type Customer struct {
	// AlternateEmail: The alternate email of the customer.
	AlternateEmail string `json:"alternateEmail,omitempty"`

	// CustomerDomain: The domain name of the customer.
	CustomerDomain string `json:"customerDomain,omitempty"`

	// CustomerDomainVerified: Whether the customer's primary domain has
	// been verified.
	CustomerDomainVerified bool `json:"customerDomainVerified,omitempty"`

	// CustomerId: The id of the customer.
	CustomerId string `json:"customerId,omitempty"`

	// Kind: Identifies the resource as a customer.
	Kind string `json:"kind,omitempty"`

	// PhoneNumber: The phone number of the customer.
	PhoneNumber string `json:"phoneNumber,omitempty"`

	// PostalAddress: The postal address of the customer.
	PostalAddress *Address `json:"postalAddress,omitempty"`

	// ResourceUiUrl: Ui url for customer resource.
	ResourceUiUrl string `json:"resourceUiUrl,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AlternateEmail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Customer) MarshalJSON() ([]byte, error) {
	type noMethod Customer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// RenewalSettings: JSON template for a subscription renewal settings.
type RenewalSettings struct {
	// Kind: Identifies the resource as a subscription renewal setting.
	Kind string `json:"kind,omitempty"`

	// RenewalType: Subscription renewal type.
	RenewalType string `json:"renewalType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *RenewalSettings) MarshalJSON() ([]byte, error) {
	type noMethod RenewalSettings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Seats: JSON template for subscription seats.
type Seats struct {
	// Kind: Identifies the resource as a subscription change plan request.
	Kind string `json:"kind,omitempty"`

	// LicensedNumberOfSeats: Read-only field containing the current number
	// of licensed seats for FLEXIBLE Google-Apps subscriptions and
	// secondary subscriptions such as Google-Vault and Drive-storage.
	LicensedNumberOfSeats int64 `json:"licensedNumberOfSeats,omitempty"`

	// MaximumNumberOfSeats: Maximum number of seats that can be purchased.
	// This needs to be provided only for a non-commitment plan. For a
	// commitment plan it is decided by the contract.
	MaximumNumberOfSeats int64 `json:"maximumNumberOfSeats,omitempty"`

	// NumberOfSeats: Number of seats to purchase. This is applicable only
	// for a commitment plan.
	NumberOfSeats int64 `json:"numberOfSeats,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Seats) MarshalJSON() ([]byte, error) {
	type noMethod Seats
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Subscription: JSON template for a subscription.
type Subscription struct {
	// BillingMethod: Billing method of this subscription.
	BillingMethod string `json:"billingMethod,omitempty"`

	// CreationTime: Creation time of this subscription in milliseconds
	// since Unix epoch.
	CreationTime int64 `json:"creationTime,omitempty,string"`

	// CustomerId: The id of the customer to whom the subscription belongs.
	CustomerId string `json:"customerId,omitempty"`

	// Kind: Identifies the resource as a Subscription.
	Kind string `json:"kind,omitempty"`

	// Plan: Plan details of the subscription
	Plan *SubscriptionPlan `json:"plan,omitempty"`

	// PurchaseOrderId: Purchase order id for your order tracking purposes.
	PurchaseOrderId string `json:"purchaseOrderId,omitempty"`

	// RenewalSettings: Renewal settings of the subscription.
	RenewalSettings *RenewalSettings `json:"renewalSettings,omitempty"`

	// ResourceUiUrl: Ui url for subscription resource.
	ResourceUiUrl string `json:"resourceUiUrl,omitempty"`

	// Seats: Number/Limit of seats in the new plan.
	Seats *Seats `json:"seats,omitempty"`

	// SkuId: Name of the sku for which this subscription is purchased.
	SkuId string `json:"skuId,omitempty"`

	// Status: Status of the subscription.
	Status string `json:"status,omitempty"`

	// SubscriptionId: The id of the subscription.
	SubscriptionId string `json:"subscriptionId,omitempty"`

	// SuspensionReasons: field listing all current reasons the subscription
	// is suspended. It is possible for a subscription to have multiple
	// suspension reasons. A subscription's status is SUSPENDED until all
	// pending suspensions are removed. Possible options include:
	// - PENDING_TOS_ACCEPTANCE — The customer has not logged in and
	// accepted the Google Apps Resold Terms of Services.
	// - RENEWAL_WITH_TYPE_CANCEL — The customer's commitment ended and
	// their service was cancelled at the end of their term.
	// - RESELLER_INITIATED — A manual suspension invoked by a Reseller.
	//
	// - TRIAL_ENDED — The customer's trial expired without a plan
	// selected.
	// - OTHER — The customer is suspended for an internal Google reason
	// (e.g. abuse or otherwise).
	SuspensionReasons []string `json:"suspensionReasons,omitempty"`

	// TransferInfo: Transfer related information for the subscription.
	TransferInfo *SubscriptionTransferInfo `json:"transferInfo,omitempty"`

	// TrialSettings: Trial Settings of the subscription.
	TrialSettings *SubscriptionTrialSettings `json:"trialSettings,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BillingMethod") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Subscription) MarshalJSON() ([]byte, error) {
	type noMethod Subscription
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SubscriptionPlan: Plan details of the subscription
type SubscriptionPlan struct {
	// CommitmentInterval: Interval of the commitment if it is a commitment
	// plan.
	CommitmentInterval *SubscriptionPlanCommitmentInterval `json:"commitmentInterval,omitempty"`

	// IsCommitmentPlan: Whether the plan is a commitment plan or not.
	IsCommitmentPlan bool `json:"isCommitmentPlan,omitempty"`

	// PlanName: The plan name of this subscription's plan.
	PlanName string `json:"planName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommitmentInterval")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionPlan) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionPlan
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SubscriptionPlanCommitmentInterval: Interval of the commitment if it
// is a commitment plan.
type SubscriptionPlanCommitmentInterval struct {
	// EndTime: End time of the commitment interval in milliseconds since
	// Unix epoch.
	EndTime int64 `json:"endTime,omitempty,string"`

	// StartTime: Start time of the commitment interval in milliseconds
	// since Unix epoch.
	StartTime int64 `json:"startTime,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "EndTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionPlanCommitmentInterval) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionPlanCommitmentInterval
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SubscriptionTransferInfo: Transfer related information for the
// subscription.
type SubscriptionTransferInfo struct {
	MinimumTransferableSeats int64 `json:"minimumTransferableSeats,omitempty"`

	// TransferabilityExpirationTime: Time when transfer token or intent to
	// transfer will expire.
	TransferabilityExpirationTime int64 `json:"transferabilityExpirationTime,omitempty,string"`

	// ForceSendFields is a list of field names (e.g.
	// "MinimumTransferableSeats") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionTransferInfo) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionTransferInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SubscriptionTrialSettings: Trial Settings of the subscription.
type SubscriptionTrialSettings struct {
	// IsInTrial: Whether the subscription is in trial.
	IsInTrial bool `json:"isInTrial,omitempty"`

	// TrialEndTime: End time of the trial in milliseconds since Unix epoch.
	TrialEndTime int64 `json:"trialEndTime,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "IsInTrial") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubscriptionTrialSettings) MarshalJSON() ([]byte, error) {
	type noMethod SubscriptionTrialSettings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Subscriptions: JSON template for a subscription list.
type Subscriptions struct {
	// Kind: Identifies the resource as a collection of subscriptions.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Subscriptions: The subscriptions in this page of results.
	Subscriptions []*Subscription `json:"subscriptions,omitempty"`

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

func (s *Subscriptions) MarshalJSON() ([]byte, error) {
	type noMethod Subscriptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "reseller.customers.get":

type CustomersGetCall struct {
	s            *Service
	customerId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets a customer resource if one exists and is owned by the
// reseller.
func (r *CustomersService) Get(customerId string) *CustomersGetCall {
	c := &CustomersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CustomersGetCall) QuotaUser(quotaUser string) *CustomersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CustomersGetCall) UserIP(userIP string) *CustomersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CustomersGetCall) Fields(s ...googleapi.Field) *CustomersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CustomersGetCall) IfNoneMatch(entityTag string) *CustomersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CustomersGetCall) Context(ctx context.Context) *CustomersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CustomersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId": c.customerId,
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

// Do executes the "reseller.customers.get" call.
// Exactly one of *Customer or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Customer.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CustomersGetCall) Do() (*Customer, error) {
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
	ret := &Customer{
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
	//   "description": "Gets a customer resource if one exists and is owned by the reseller.",
	//   "httpMethod": "GET",
	//   "id": "reseller.customers.get",
	//   "parameterOrder": [
	//     "customerId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}",
	//   "response": {
	//     "$ref": "Customer"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order",
	//     "https://www.googleapis.com/auth/apps.order.readonly"
	//   ]
	// }

}

// method id "reseller.customers.insert":

type CustomersInsertCall struct {
	s          *Service
	customer   *Customer
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Creates a customer resource if one does not already exist.
func (r *CustomersService) Insert(customer *Customer) *CustomersInsertCall {
	c := &CustomersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customer = customer
	return c
}

// CustomerAuthToken sets the optional parameter "customerAuthToken": An
// auth token needed for inserting a customer for which domain already
// exists. Can be generated at
// https://www.google.com/a/cpanel//TransferToken.
func (c *CustomersInsertCall) CustomerAuthToken(customerAuthToken string) *CustomersInsertCall {
	c.urlParams_.Set("customerAuthToken", customerAuthToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CustomersInsertCall) QuotaUser(quotaUser string) *CustomersInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CustomersInsertCall) UserIP(userIP string) *CustomersInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CustomersInsertCall) Fields(s ...googleapi.Field) *CustomersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CustomersInsertCall) Context(ctx context.Context) *CustomersInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *CustomersInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.customer)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers")
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

// Do executes the "reseller.customers.insert" call.
// Exactly one of *Customer or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Customer.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CustomersInsertCall) Do() (*Customer, error) {
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
	ret := &Customer{
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
	//   "description": "Creates a customer resource if one does not already exist.",
	//   "httpMethod": "POST",
	//   "id": "reseller.customers.insert",
	//   "parameters": {
	//     "customerAuthToken": {
	//       "description": "An auth token needed for inserting a customer for which domain already exists. Can be generated at https://www.google.com/a/cpanel//TransferToken. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers",
	//   "request": {
	//     "$ref": "Customer"
	//   },
	//   "response": {
	//     "$ref": "Customer"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.customers.patch":

type CustomersPatchCall struct {
	s          *Service
	customerId string
	customer   *Customer
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Update a customer resource if one it exists and is owned by
// the reseller. This method supports patch semantics.
func (r *CustomersService) Patch(customerId string, customer *Customer) *CustomersPatchCall {
	c := &CustomersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.customer = customer
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CustomersPatchCall) QuotaUser(quotaUser string) *CustomersPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CustomersPatchCall) UserIP(userIP string) *CustomersPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CustomersPatchCall) Fields(s ...googleapi.Field) *CustomersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CustomersPatchCall) Context(ctx context.Context) *CustomersPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *CustomersPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.customer)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId": c.customerId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.customers.patch" call.
// Exactly one of *Customer or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Customer.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CustomersPatchCall) Do() (*Customer, error) {
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
	ret := &Customer{
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
	//   "description": "Update a customer resource if one it exists and is owned by the reseller. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "reseller.customers.patch",
	//   "parameterOrder": [
	//     "customerId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}",
	//   "request": {
	//     "$ref": "Customer"
	//   },
	//   "response": {
	//     "$ref": "Customer"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.customers.update":

type CustomersUpdateCall struct {
	s          *Service
	customerId string
	customer   *Customer
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Update a customer resource if one it exists and is owned by
// the reseller.
func (r *CustomersService) Update(customerId string, customer *Customer) *CustomersUpdateCall {
	c := &CustomersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.customer = customer
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CustomersUpdateCall) QuotaUser(quotaUser string) *CustomersUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CustomersUpdateCall) UserIP(userIP string) *CustomersUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CustomersUpdateCall) Fields(s ...googleapi.Field) *CustomersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CustomersUpdateCall) Context(ctx context.Context) *CustomersUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *CustomersUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.customer)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId": c.customerId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.customers.update" call.
// Exactly one of *Customer or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Customer.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CustomersUpdateCall) Do() (*Customer, error) {
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
	ret := &Customer{
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
	//   "description": "Update a customer resource if one it exists and is owned by the reseller.",
	//   "httpMethod": "PUT",
	//   "id": "reseller.customers.update",
	//   "parameterOrder": [
	//     "customerId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}",
	//   "request": {
	//     "$ref": "Customer"
	//   },
	//   "response": {
	//     "$ref": "Customer"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.activate":

type SubscriptionsActivateCall struct {
	s              *Service
	customerId     string
	subscriptionId string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Activate: Activates a subscription previously suspended by the
// reseller
func (r *SubscriptionsService) Activate(customerId string, subscriptionId string) *SubscriptionsActivateCall {
	c := &SubscriptionsActivateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsActivateCall) QuotaUser(quotaUser string) *SubscriptionsActivateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsActivateCall) UserIP(userIP string) *SubscriptionsActivateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsActivateCall) Fields(s ...googleapi.Field) *SubscriptionsActivateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsActivateCall) Context(ctx context.Context) *SubscriptionsActivateCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsActivateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/activate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.activate" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsActivateCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Activates a subscription previously suspended by the reseller",
	//   "httpMethod": "POST",
	//   "id": "reseller.subscriptions.activate",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}/activate",
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.changePlan":

type SubscriptionsChangePlanCall struct {
	s                 *Service
	customerId        string
	subscriptionId    string
	changeplanrequest *ChangePlanRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
}

// ChangePlan: Changes the plan of a subscription
func (r *SubscriptionsService) ChangePlan(customerId string, subscriptionId string, changeplanrequest *ChangePlanRequest) *SubscriptionsChangePlanCall {
	c := &SubscriptionsChangePlanCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.changeplanrequest = changeplanrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsChangePlanCall) QuotaUser(quotaUser string) *SubscriptionsChangePlanCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsChangePlanCall) UserIP(userIP string) *SubscriptionsChangePlanCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsChangePlanCall) Fields(s ...googleapi.Field) *SubscriptionsChangePlanCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsChangePlanCall) Context(ctx context.Context) *SubscriptionsChangePlanCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsChangePlanCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.changeplanrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/changePlan")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.changePlan" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsChangePlanCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Changes the plan of a subscription",
	//   "httpMethod": "POST",
	//   "id": "reseller.subscriptions.changePlan",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}/changePlan",
	//   "request": {
	//     "$ref": "ChangePlanRequest"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.changeRenewalSettings":

type SubscriptionsChangeRenewalSettingsCall struct {
	s               *Service
	customerId      string
	subscriptionId  string
	renewalsettings *RenewalSettings
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// ChangeRenewalSettings: Changes the renewal settings of a subscription
func (r *SubscriptionsService) ChangeRenewalSettings(customerId string, subscriptionId string, renewalsettings *RenewalSettings) *SubscriptionsChangeRenewalSettingsCall {
	c := &SubscriptionsChangeRenewalSettingsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.renewalsettings = renewalsettings
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsChangeRenewalSettingsCall) QuotaUser(quotaUser string) *SubscriptionsChangeRenewalSettingsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsChangeRenewalSettingsCall) UserIP(userIP string) *SubscriptionsChangeRenewalSettingsCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsChangeRenewalSettingsCall) Fields(s ...googleapi.Field) *SubscriptionsChangeRenewalSettingsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsChangeRenewalSettingsCall) Context(ctx context.Context) *SubscriptionsChangeRenewalSettingsCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsChangeRenewalSettingsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.renewalsettings)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/changeRenewalSettings")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.changeRenewalSettings" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsChangeRenewalSettingsCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Changes the renewal settings of a subscription",
	//   "httpMethod": "POST",
	//   "id": "reseller.subscriptions.changeRenewalSettings",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}/changeRenewalSettings",
	//   "request": {
	//     "$ref": "RenewalSettings"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.changeSeats":

type SubscriptionsChangeSeatsCall struct {
	s              *Service
	customerId     string
	subscriptionId string
	seats          *Seats
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// ChangeSeats: Changes the seats configuration of a subscription
func (r *SubscriptionsService) ChangeSeats(customerId string, subscriptionId string, seats *Seats) *SubscriptionsChangeSeatsCall {
	c := &SubscriptionsChangeSeatsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.seats = seats
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsChangeSeatsCall) QuotaUser(quotaUser string) *SubscriptionsChangeSeatsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsChangeSeatsCall) UserIP(userIP string) *SubscriptionsChangeSeatsCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsChangeSeatsCall) Fields(s ...googleapi.Field) *SubscriptionsChangeSeatsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsChangeSeatsCall) Context(ctx context.Context) *SubscriptionsChangeSeatsCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsChangeSeatsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.seats)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/changeSeats")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.changeSeats" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsChangeSeatsCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Changes the seats configuration of a subscription",
	//   "httpMethod": "POST",
	//   "id": "reseller.subscriptions.changeSeats",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}/changeSeats",
	//   "request": {
	//     "$ref": "Seats"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.delete":

type SubscriptionsDeleteCall struct {
	s              *Service
	customerId     string
	subscriptionId string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Delete: Cancels/Downgrades a subscription.
func (r *SubscriptionsService) Delete(customerId string, subscriptionId string, deletionType string) *SubscriptionsDeleteCall {
	c := &SubscriptionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.urlParams_.Set("deletionType", deletionType)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsDeleteCall) QuotaUser(quotaUser string) *SubscriptionsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsDeleteCall) UserIP(userIP string) *SubscriptionsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsDeleteCall) Fields(s ...googleapi.Field) *SubscriptionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsDeleteCall) Context(ctx context.Context) *SubscriptionsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.delete" call.
func (c *SubscriptionsDeleteCall) Do() error {
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
	//   "description": "Cancels/Downgrades a subscription.",
	//   "httpMethod": "DELETE",
	//   "id": "reseller.subscriptions.delete",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId",
	//     "deletionType"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "deletionType": {
	//       "description": "Whether the subscription is to be fully cancelled or downgraded",
	//       "enum": [
	//         "cancel",
	//         "downgrade",
	//         "suspend",
	//         "transfer_to_direct"
	//       ],
	//       "enumDescriptions": [
	//         "Cancels the subscription immediately",
	//         "Downgrades a Google Apps for Business subscription to Google Apps",
	//         "Suspends the subscriptions for 4 days before cancelling it",
	//         "Transfers a subscription directly to Google"
	//       ],
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.get":

type SubscriptionsGetCall struct {
	s              *Service
	customerId     string
	subscriptionId string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
}

// Get: Gets a subscription of the customer.
func (r *SubscriptionsService) Get(customerId string, subscriptionId string) *SubscriptionsGetCall {
	c := &SubscriptionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsGetCall) QuotaUser(quotaUser string) *SubscriptionsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsGetCall) UserIP(userIP string) *SubscriptionsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsGetCall) Fields(s ...googleapi.Field) *SubscriptionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SubscriptionsGetCall) IfNoneMatch(entityTag string) *SubscriptionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsGetCall) Context(ctx context.Context) *SubscriptionsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
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

// Do executes the "reseller.subscriptions.get" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsGetCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Gets a subscription of the customer.",
	//   "httpMethod": "GET",
	//   "id": "reseller.subscriptions.get",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}",
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order",
	//     "https://www.googleapis.com/auth/apps.order.readonly"
	//   ]
	// }

}

// method id "reseller.subscriptions.insert":

type SubscriptionsInsertCall struct {
	s            *Service
	customerId   string
	subscription *Subscription
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Insert: Creates/Transfers a subscription for the customer.
func (r *SubscriptionsService) Insert(customerId string, subscription *Subscription) *SubscriptionsInsertCall {
	c := &SubscriptionsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscription = subscription
	return c
}

// CustomerAuthToken sets the optional parameter "customerAuthToken": An
// auth token needed for transferring a subscription. Can be generated
// at https://www.google.com/a/cpanel/customer-domain/TransferToken.
func (c *SubscriptionsInsertCall) CustomerAuthToken(customerAuthToken string) *SubscriptionsInsertCall {
	c.urlParams_.Set("customerAuthToken", customerAuthToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsInsertCall) QuotaUser(quotaUser string) *SubscriptionsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsInsertCall) UserIP(userIP string) *SubscriptionsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsInsertCall) Fields(s ...googleapi.Field) *SubscriptionsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsInsertCall) Context(ctx context.Context) *SubscriptionsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.subscription)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId": c.customerId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.insert" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsInsertCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Creates/Transfers a subscription for the customer.",
	//   "httpMethod": "POST",
	//   "id": "reseller.subscriptions.insert",
	//   "parameterOrder": [
	//     "customerId"
	//   ],
	//   "parameters": {
	//     "customerAuthToken": {
	//       "description": "An auth token needed for transferring a subscription. Can be generated at https://www.google.com/a/cpanel/customer-domain/TransferToken. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions",
	//   "request": {
	//     "$ref": "Subscription"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.list":

type SubscriptionsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists subscriptions of a reseller, optionally filtered by a
// customer name prefix.
func (r *SubscriptionsService) List() *SubscriptionsListCall {
	c := &SubscriptionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// CustomerAuthToken sets the optional parameter "customerAuthToken": An
// auth token needed if the customer is not a resold customer of this
// reseller. Can be generated at
// https://www.google.com/a/cpanel/customer-domain/TransferToken.
func (c *SubscriptionsListCall) CustomerAuthToken(customerAuthToken string) *SubscriptionsListCall {
	c.urlParams_.Set("customerAuthToken", customerAuthToken)
	return c
}

// CustomerId sets the optional parameter "customerId": Id of the
// Customer
func (c *SubscriptionsListCall) CustomerId(customerId string) *SubscriptionsListCall {
	c.urlParams_.Set("customerId", customerId)
	return c
}

// CustomerNamePrefix sets the optional parameter "customerNamePrefix":
// Prefix of the customer's domain name by which the subscriptions
// should be filtered. Optional
func (c *SubscriptionsListCall) CustomerNamePrefix(customerNamePrefix string) *SubscriptionsListCall {
	c.urlParams_.Set("customerNamePrefix", customerNamePrefix)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *SubscriptionsListCall) MaxResults(maxResults int64) *SubscriptionsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token to specify
// next page in the list
func (c *SubscriptionsListCall) PageToken(pageToken string) *SubscriptionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsListCall) QuotaUser(quotaUser string) *SubscriptionsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsListCall) UserIP(userIP string) *SubscriptionsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsListCall) Fields(s ...googleapi.Field) *SubscriptionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SubscriptionsListCall) IfNoneMatch(entityTag string) *SubscriptionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsListCall) Context(ctx context.Context) *SubscriptionsListCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "subscriptions")
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

// Do executes the "reseller.subscriptions.list" call.
// Exactly one of *Subscriptions or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscriptions.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SubscriptionsListCall) Do() (*Subscriptions, error) {
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
	ret := &Subscriptions{
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
	//   "description": "Lists subscriptions of a reseller, optionally filtered by a customer name prefix.",
	//   "httpMethod": "GET",
	//   "id": "reseller.subscriptions.list",
	//   "parameters": {
	//     "customerAuthToken": {
	//       "description": "An auth token needed if the customer is not a resold customer of this reseller. Can be generated at https://www.google.com/a/cpanel/customer-domain/TransferToken.Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customerNamePrefix": {
	//       "description": "Prefix of the customer's domain name by which the subscriptions should be filtered. Optional",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Token to specify next page in the list",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "subscriptions",
	//   "response": {
	//     "$ref": "Subscriptions"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order",
	//     "https://www.googleapis.com/auth/apps.order.readonly"
	//   ]
	// }

}

// method id "reseller.subscriptions.startPaidService":

type SubscriptionsStartPaidServiceCall struct {
	s              *Service
	customerId     string
	subscriptionId string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// StartPaidService: Starts paid service of a trial subscription
func (r *SubscriptionsService) StartPaidService(customerId string, subscriptionId string) *SubscriptionsStartPaidServiceCall {
	c := &SubscriptionsStartPaidServiceCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsStartPaidServiceCall) QuotaUser(quotaUser string) *SubscriptionsStartPaidServiceCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsStartPaidServiceCall) UserIP(userIP string) *SubscriptionsStartPaidServiceCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsStartPaidServiceCall) Fields(s ...googleapi.Field) *SubscriptionsStartPaidServiceCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsStartPaidServiceCall) Context(ctx context.Context) *SubscriptionsStartPaidServiceCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsStartPaidServiceCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/startPaidService")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.startPaidService" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsStartPaidServiceCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Starts paid service of a trial subscription",
	//   "httpMethod": "POST",
	//   "id": "reseller.subscriptions.startPaidService",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}/startPaidService",
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}

// method id "reseller.subscriptions.suspend":

type SubscriptionsSuspendCall struct {
	s              *Service
	customerId     string
	subscriptionId string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Suspend: Suspends an active subscription
func (r *SubscriptionsService) Suspend(customerId string, subscriptionId string) *SubscriptionsSuspendCall {
	c := &SubscriptionsSuspendCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SubscriptionsSuspendCall) QuotaUser(quotaUser string) *SubscriptionsSuspendCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SubscriptionsSuspendCall) UserIP(userIP string) *SubscriptionsSuspendCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsSuspendCall) Fields(s ...googleapi.Field) *SubscriptionsSuspendCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsSuspendCall) Context(ctx context.Context) *SubscriptionsSuspendCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsSuspendCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/suspend")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"customerId":     c.customerId,
		"subscriptionId": c.subscriptionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "reseller.subscriptions.suspend" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsSuspendCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Suspends an active subscription",
	//   "httpMethod": "POST",
	//   "id": "reseller.subscriptions.suspend",
	//   "parameterOrder": [
	//     "customerId",
	//     "subscriptionId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Id of the Customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "Id of the subscription, which is unique for a customer",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "customers/{customerId}/subscriptions/{subscriptionId}/suspend",
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.order"
	//   ]
	// }

}
