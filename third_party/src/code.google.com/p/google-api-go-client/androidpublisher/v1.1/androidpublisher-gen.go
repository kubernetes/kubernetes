// Package androidpublisher provides access to the Google Play Android Developer API.
//
// See https://developers.google.com/android-publisher
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/androidpublisher/v1.1"
//   ...
//   androidpublisherService, err := androidpublisher.New(oauthHttpClient)
package androidpublisher

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

const apiId = "androidpublisher:v1.1"
const apiName = "androidpublisher"
const apiVersion = "v1.1"
const basePath = "https://www.googleapis.com/androidpublisher/v1.1/applications/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Google Play Android Developer account
	AndroidpublisherScope = "https://www.googleapis.com/auth/androidpublisher"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Inapppurchases = NewInapppurchasesService(s)
	s.Purchases = NewPurchasesService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Inapppurchases *InapppurchasesService

	Purchases *PurchasesService
}

func NewInapppurchasesService(s *Service) *InapppurchasesService {
	rs := &InapppurchasesService{s: s}
	return rs
}

type InapppurchasesService struct {
	s *Service
}

func NewPurchasesService(s *Service) *PurchasesService {
	rs := &PurchasesService{s: s}
	return rs
}

type PurchasesService struct {
	s *Service
}

type InappPurchase struct {
	// ConsumptionState: The consumption state of the inapp product.
	// Possible values are:
	// - Yet to be consumed
	// - Consumed
	ConsumptionState int64 `json:"consumptionState,omitempty"`

	// DeveloperPayload: A developer-specified string that contains
	// supplemental information about an order.
	DeveloperPayload string `json:"developerPayload,omitempty"`

	// Kind: This kind represents a inappPurchase object in the
	// androidpublisher service.
	Kind string `json:"kind,omitempty"`

	// PurchaseState: The purchase state of the order. Possible values are:
	//
	// - Purchased
	// - Cancelled
	PurchaseState int64 `json:"purchaseState,omitempty"`

	// PurchaseTime: The time the product was purchased, in milliseconds
	// since the epoch (Jan 1, 1970).
	PurchaseTime int64 `json:"purchaseTime,omitempty,string"`
}

type SubscriptionPurchase struct {
	// AutoRenewing: Whether the subscription will automatically be renewed
	// when it reaches its current expiry time.
	AutoRenewing bool `json:"autoRenewing,omitempty"`

	// InitiationTimestampMsec: Time at which the subscription was granted,
	// in milliseconds since Epoch.
	InitiationTimestampMsec int64 `json:"initiationTimestampMsec,omitempty,string"`

	// Kind: This kind represents a subscriptionPurchase object in the
	// androidpublisher service.
	Kind string `json:"kind,omitempty"`

	// ValidUntilTimestampMsec: Time at which the subscription will expire,
	// in milliseconds since Epoch.
	ValidUntilTimestampMsec int64 `json:"validUntilTimestampMsec,omitempty,string"`
}

// method id "androidpublisher.inapppurchases.get":

type InapppurchasesGetCall struct {
	s           *Service
	packageName string
	productId   string
	token       string
	opt_        map[string]interface{}
}

// Get: Checks the purchase and consumption status of an inapp item.
func (r *InapppurchasesService) Get(packageName string, productId string, token string) *InapppurchasesGetCall {
	c := &InapppurchasesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.packageName = packageName
	c.productId = productId
	c.token = token
	return c
}

func (c *InapppurchasesGetCall) Do() (*InappPurchase, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/inapp/{productId}/purchases/{token}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{packageName}", url.QueryEscape(c.packageName), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{token}", url.QueryEscape(c.token), 1)
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
	ret := new(InappPurchase)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Checks the purchase and consumption status of an inapp item.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.inapppurchases.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "productId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application the inapp product was sold in (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The inapp product SKU (for example, 'com.some.thing.inapp1').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the inapp product was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/inapp/{productId}/purchases/{token}",
	//   "response": {
	//     "$ref": "InappPurchase"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.cancel":

type PurchasesCancelCall struct {
	s              *Service
	packageName    string
	subscriptionId string
	token          string
	opt_           map[string]interface{}
}

// Cancel: Cancels a user's subscription purchase. The subscription
// remains valid until its expiration time.
func (r *PurchasesService) Cancel(packageName string, subscriptionId string, token string) *PurchasesCancelCall {
	c := &PurchasesCancelCall{s: r.s, opt_: make(map[string]interface{})}
	c.packageName = packageName
	c.subscriptionId = subscriptionId
	c.token = token
	return c
}

func (c *PurchasesCancelCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/subscriptions/{subscriptionId}/purchases/{token}/cancel")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{packageName}", url.QueryEscape(c.packageName), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{token}", url.QueryEscape(c.token), 1)
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
	//   "description": "Cancels a user's subscription purchase. The subscription remains valid until its expiration time.",
	//   "httpMethod": "POST",
	//   "id": "androidpublisher.purchases.cancel",
	//   "parameterOrder": [
	//     "packageName",
	//     "subscriptionId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application for which this subscription was purchased (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "The purchased subscription ID (for example, 'monthly001').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the subscription was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/subscriptions/{subscriptionId}/purchases/{token}/cancel",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}

// method id "androidpublisher.purchases.get":

type PurchasesGetCall struct {
	s              *Service
	packageName    string
	subscriptionId string
	token          string
	opt_           map[string]interface{}
}

// Get: Checks whether a user's subscription purchase is valid and
// returns its expiry time.
func (r *PurchasesService) Get(packageName string, subscriptionId string, token string) *PurchasesGetCall {
	c := &PurchasesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.packageName = packageName
	c.subscriptionId = subscriptionId
	c.token = token
	return c
}

func (c *PurchasesGetCall) Do() (*SubscriptionPurchase, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{packageName}/subscriptions/{subscriptionId}/purchases/{token}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{packageName}", url.QueryEscape(c.packageName), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{subscriptionId}", url.QueryEscape(c.subscriptionId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{token}", url.QueryEscape(c.token), 1)
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
	ret := new(SubscriptionPurchase)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Checks whether a user's subscription purchase is valid and returns its expiry time.",
	//   "httpMethod": "GET",
	//   "id": "androidpublisher.purchases.get",
	//   "parameterOrder": [
	//     "packageName",
	//     "subscriptionId",
	//     "token"
	//   ],
	//   "parameters": {
	//     "packageName": {
	//       "description": "The package name of the application for which this subscription was purchased (for example, 'com.some.thing').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "subscriptionId": {
	//       "description": "The purchased subscription ID (for example, 'monthly001').",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "The token provided to the user's device when the subscription was purchased.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{packageName}/subscriptions/{subscriptionId}/purchases/{token}",
	//   "response": {
	//     "$ref": "SubscriptionPurchase"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidpublisher"
	//   ]
	// }

}
