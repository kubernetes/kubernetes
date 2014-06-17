// Package reseller provides access to the Enterprise Apps Reseller API.
//
// See https://developers.google.com/google-apps/reseller/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/reseller/v1sandbox"
//   ...
//   resellerService, err := reseller.New(oauthHttpClient)
package reseller

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

const apiId = "reseller:v1sandbox"
const apiName = "reseller"
const apiVersion = "v1sandbox"
const basePath = "https://www.googleapis.com/apps/reseller/v1sandbox/"

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
	client   *http.Client
	BasePath string // API endpoint base URL

	Customers *CustomersService

	Subscriptions *SubscriptionsService
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
}

type ChangePlanRequest struct {
	// Kind: Identifies the resource as a subscription change plan request.
	Kind string `json:"kind,omitempty"`

	// PlanName: Name of the plan to change to.
	PlanName string `json:"planName,omitempty"`

	// PurchaseOrderId: Purchase order id for your order tracking purposes.
	PurchaseOrderId string `json:"purchaseOrderId,omitempty"`

	// Seats: Number/Limit of seats in the new plan.
	Seats *Seats `json:"seats,omitempty"`
}

type Customer struct {
	// AlternateEmail: The alternate email of the customer.
	AlternateEmail string `json:"alternateEmail,omitempty"`

	// CustomerDomain: The domain name of the customer.
	CustomerDomain string `json:"customerDomain,omitempty"`

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
}

type RenewalSettings struct {
	// Kind: Identifies the resource as a subscription renewal setting.
	Kind string `json:"kind,omitempty"`

	// RenewalType: Subscription renewal type.
	RenewalType string `json:"renewalType,omitempty"`
}

type Seats struct {
	// Kind: Identifies the resource as a subscription change plan request.
	Kind string `json:"kind,omitempty"`

	// MaximumNumberOfSeats: Maximum number of seats that can be purchased.
	// This needs to be provided only for a non-commitment plan. For a
	// commitment plan it is decided by the contract.
	MaximumNumberOfSeats int64 `json:"maximumNumberOfSeats,omitempty"`

	// NumberOfSeats: Number of seats to purchase. This is applicable only
	// for a commitment plan.
	NumberOfSeats int64 `json:"numberOfSeats,omitempty"`
}

type Subscription struct {
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

	// TransferInfo: Transfer related information for the subscription.
	TransferInfo *SubscriptionTransferInfo `json:"transferInfo,omitempty"`

	// TrialSettings: Trial Settings of the subscription.
	TrialSettings *SubscriptionTrialSettings `json:"trialSettings,omitempty"`
}

type SubscriptionPlan struct {
	// CommitmentInterval: Interval of the commitment if it is a commitment
	// plan.
	CommitmentInterval *SubscriptionPlanCommitmentInterval `json:"commitmentInterval,omitempty"`

	// IsCommitmentPlan: Whether the plan is a commitment plan or not.
	IsCommitmentPlan bool `json:"isCommitmentPlan,omitempty"`

	// PlanName: The plan name of this subscription's plan.
	PlanName string `json:"planName,omitempty"`
}

type SubscriptionPlanCommitmentInterval struct {
	// EndTime: End time of the commitment interval in milliseconds since
	// Unix epoch.
	EndTime int64 `json:"endTime,omitempty,string"`

	// StartTime: Start time of the commitment interval in milliseconds
	// since Unix epoch.
	StartTime int64 `json:"startTime,omitempty,string"`
}

type SubscriptionTransferInfo struct {
	MinimumTransferableSeats int64 `json:"minimumTransferableSeats,omitempty"`

	// TransferabilityExpirationTime: Time when transfer token or intent to
	// transfer will expire.
	TransferabilityExpirationTime int64 `json:"transferabilityExpirationTime,omitempty,string"`
}

type SubscriptionTrialSettings struct {
	// IsInTrial: Whether the subscription is in trial.
	IsInTrial bool `json:"isInTrial,omitempty"`

	// TrialEndTime: End time of the trial in milliseconds since Unix epoch.
	TrialEndTime int64 `json:"trialEndTime,omitempty,string"`
}

type Subscriptions struct {
	// Kind: Identifies the resource as a collection of subscriptions.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Subscriptions: The subscriptions in this page of results.
	Subscriptions []*Subscription `json:"subscriptions,omitempty"`
}

// method id "reseller.customers.get":

type CustomersGetCall struct {
	s          *Service
	customerId string
	opt_       map[string]interface{}
}

// Get: Gets a customer resource if one exists and is owned by the
// reseller.
func (r *CustomersService) Get(customerId string) *CustomersGetCall {
	c := &CustomersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	return c
}

func (c *CustomersGetCall) Do() (*Customer, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
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
	ret := new(Customer)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	s        *Service
	customer *Customer
	opt_     map[string]interface{}
}

// Insert: Creates a customer resource if one does not already exist.
func (r *CustomersService) Insert(customer *Customer) *CustomersInsertCall {
	c := &CustomersInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.customer = customer
	return c
}

// CustomerAuthToken sets the optional parameter "customerAuthToken": An
// auth token needed for inserting a customer for which domain already
// exists. Can be generated at
// https://www.google.com/a/cpanel//TransferToken.
func (c *CustomersInsertCall) CustomerAuthToken(customerAuthToken string) *CustomersInsertCall {
	c.opt_["customerAuthToken"] = customerAuthToken
	return c
}

func (c *CustomersInsertCall) Do() (*Customer, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.customer)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["customerAuthToken"]; ok {
		params.Set("customerAuthToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers")
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
	ret := new(Customer)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_       map[string]interface{}
}

// Patch: Update a customer resource if one it exists and is owned by
// the reseller. This method supports patch semantics.
func (r *CustomersService) Patch(customerId string, customer *Customer) *CustomersPatchCall {
	c := &CustomersPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.customer = customer
	return c
}

func (c *CustomersPatchCall) Do() (*Customer, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.customer)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
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
	ret := new(Customer)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_       map[string]interface{}
}

// Update: Update a customer resource if one it exists and is owned by
// the reseller.
func (r *CustomersService) Update(customerId string, customer *Customer) *CustomersUpdateCall {
	c := &CustomersUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.customer = customer
	return c
}

func (c *CustomersUpdateCall) Do() (*Customer, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.customer)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
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
	ret := new(Customer)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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

// method id "reseller.subscriptions.changePlan":

type SubscriptionsChangePlanCall struct {
	s                 *Service
	customerId        string
	subscriptionId    string
	changeplanrequest *ChangePlanRequest
	opt_              map[string]interface{}
}

// ChangePlan: Changes the plan of a subscription
func (r *SubscriptionsService) ChangePlan(customerId string, subscriptionId string, changeplanrequest *ChangePlanRequest) *SubscriptionsChangePlanCall {
	c := &SubscriptionsChangePlanCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.changeplanrequest = changeplanrequest
	return c
}

func (c *SubscriptionsChangePlanCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.changeplanrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/changePlan")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_            map[string]interface{}
}

// ChangeRenewalSettings: Changes the renewal settings of a subscription
func (r *SubscriptionsService) ChangeRenewalSettings(customerId string, subscriptionId string, renewalsettings *RenewalSettings) *SubscriptionsChangeRenewalSettingsCall {
	c := &SubscriptionsChangeRenewalSettingsCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.renewalsettings = renewalsettings
	return c
}

func (c *SubscriptionsChangeRenewalSettingsCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.renewalsettings)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/changeRenewalSettings")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_           map[string]interface{}
}

// ChangeSeats: Changes the seats configuration of a subscription
func (r *SubscriptionsService) ChangeSeats(customerId string, subscriptionId string, seats *Seats) *SubscriptionsChangeSeatsCall {
	c := &SubscriptionsChangeSeatsCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.seats = seats
	return c
}

func (c *SubscriptionsChangeSeatsCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.seats)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/changeSeats")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	deletionType   string
	opt_           map[string]interface{}
}

// Delete: Cancels/Downgrades a subscription.
func (r *SubscriptionsService) Delete(customerId string, subscriptionId string, deletionType string) *SubscriptionsDeleteCall {
	c := &SubscriptionsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	c.deletionType = deletionType
	return c
}

func (c *SubscriptionsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("deletionType", fmt.Sprintf("%v", c.deletionType))
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
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
	//         "suspend"
	//       ],
	//       "enumDescriptions": [
	//         "Cancels the subscription immediately",
	//         "Downgrades a Google Apps for Business subscription to Google Apps",
	//         "Suspends the subscriptions for 4 days before cancelling it"
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
	opt_           map[string]interface{}
}

// Get: Gets a subscription of the customer.
func (r *SubscriptionsService) Get(customerId string, subscriptionId string) *SubscriptionsGetCall {
	c := &SubscriptionsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	return c
}

func (c *SubscriptionsGetCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_         map[string]interface{}
}

// Insert: Creates/Transfers a subscription for the customer.
func (r *SubscriptionsService) Insert(customerId string, subscription *Subscription) *SubscriptionsInsertCall {
	c := &SubscriptionsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.subscription = subscription
	return c
}

// CustomerAuthToken sets the optional parameter "customerAuthToken": An
// auth token needed for transferring a subscription. Can be generated
// at https://www.google.com/a/cpanel/customer-domain/TransferToken.
func (c *SubscriptionsInsertCall) CustomerAuthToken(customerAuthToken string) *SubscriptionsInsertCall {
	c.opt_["customerAuthToken"] = customerAuthToken
	return c
}

func (c *SubscriptionsInsertCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.subscription)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["customerAuthToken"]; ok {
		params.Set("customerAuthToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	s    *Service
	opt_ map[string]interface{}
}

// List: Lists subscriptions of a reseller, optionally filtered by a
// customer name prefix.
func (r *SubscriptionsService) List() *SubscriptionsListCall {
	c := &SubscriptionsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// CustomerAuthToken sets the optional parameter "customerAuthToken": An
// auth token needed if the customer is not a resold customer of this
// reseller. Can be generated at
// https://www.google.com/a/cpanel/customer-domain/TransferToken.
func (c *SubscriptionsListCall) CustomerAuthToken(customerAuthToken string) *SubscriptionsListCall {
	c.opt_["customerAuthToken"] = customerAuthToken
	return c
}

// CustomerId sets the optional parameter "customerId": Id of the
// Customer
func (c *SubscriptionsListCall) CustomerId(customerId string) *SubscriptionsListCall {
	c.opt_["customerId"] = customerId
	return c
}

// CustomerNamePrefix sets the optional parameter "customerNamePrefix":
// Prefix of the customer's domain name by which the subscriptions
// should be filtered. Optional
func (c *SubscriptionsListCall) CustomerNamePrefix(customerNamePrefix string) *SubscriptionsListCall {
	c.opt_["customerNamePrefix"] = customerNamePrefix
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *SubscriptionsListCall) MaxResults(maxResults int64) *SubscriptionsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Token to specify
// next page in the list
func (c *SubscriptionsListCall) PageToken(pageToken string) *SubscriptionsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *SubscriptionsListCall) Do() (*Subscriptions, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["customerAuthToken"]; ok {
		params.Set("customerAuthToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerId"]; ok {
		params.Set("customerId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerNamePrefix"]; ok {
		params.Set("customerNamePrefix", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "subscriptions")
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
	ret := new(Subscriptions)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_           map[string]interface{}
}

// StartPaidService: Starts paid service of a trial subscription
func (r *SubscriptionsService) StartPaidService(customerId string, subscriptionId string) *SubscriptionsStartPaidServiceCall {
	c := &SubscriptionsStartPaidServiceCall{s: r.s, opt_: make(map[string]interface{})}
	c.customerId = customerId
	c.subscriptionId = subscriptionId
	return c
}

func (c *SubscriptionsStartPaidServiceCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "customers/{customerId}/subscriptions/{subscriptionId}/startPaidService")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{customerId}", url.QueryEscape(c.customerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
