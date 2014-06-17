// Package licensing provides access to the Enterprise License Manager API.
//
// See https://developers.google.com/google-apps/licensing/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/licensing/v1"
//   ...
//   licensingService, err := licensing.New(oauthHttpClient)
package licensing

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

const apiId = "licensing:v1"
const apiName = "licensing"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/apps/licensing/v1/product/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.LicenseAssignments = NewLicenseAssignmentsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	LicenseAssignments *LicenseAssignmentsService
}

func NewLicenseAssignmentsService(s *Service) *LicenseAssignmentsService {
	rs := &LicenseAssignmentsService{s: s}
	return rs
}

type LicenseAssignmentsService struct {
	s *Service
}

type LicenseAssignment struct {
	// Etags: ETag of the resource.
	Etags string `json:"etags,omitempty"`

	// Kind: Identifies the resource as a LicenseAssignment.
	Kind string `json:"kind,omitempty"`

	// ProductId: Name of the product.
	ProductId string `json:"productId,omitempty"`

	// SelfLink: Link to this page.
	SelfLink string `json:"selfLink,omitempty"`

	// SkuId: Name of the sku of the product.
	SkuId string `json:"skuId,omitempty"`

	// UserId: Email id of the user.
	UserId string `json:"userId,omitempty"`
}

type LicenseAssignmentInsert struct {
	// UserId: Email id of the user
	UserId string `json:"userId,omitempty"`
}

type LicenseAssignmentList struct {
	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Items: The LicenseAssignments in this page of results.
	Items []*LicenseAssignment `json:"items,omitempty"`

	// Kind: Identifies the resource as a collection of LicenseAssignments.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

// method id "licensing.licenseAssignments.delete":

type LicenseAssignmentsDeleteCall struct {
	s         *Service
	productId string
	skuId     string
	userId    string
	opt_      map[string]interface{}
}

// Delete: Revoke License.
func (r *LicenseAssignmentsService) Delete(productId string, skuId string, userId string) *LicenseAssignmentsDeleteCall {
	c := &LicenseAssignmentsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	return c
}

func (c *LicenseAssignmentsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{skuId}", url.QueryEscape(c.skuId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	//   "description": "Revoke License.",
	//   "httpMethod": "DELETE",
	//   "id": "licensing.licenseAssignments.delete",
	//   "parameterOrder": [
	//     "productId",
	//     "skuId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "productId": {
	//       "description": "Name for product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "skuId": {
	//       "description": "Name for sku",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "email id or unique Id of the user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{productId}/sku/{skuId}/user/{userId}"
	// }

}

// method id "licensing.licenseAssignments.get":

type LicenseAssignmentsGetCall struct {
	s         *Service
	productId string
	skuId     string
	userId    string
	opt_      map[string]interface{}
}

// Get: Get license assignment of a particular product and sku for a
// user
func (r *LicenseAssignmentsService) Get(productId string, skuId string, userId string) *LicenseAssignmentsGetCall {
	c := &LicenseAssignmentsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	return c
}

func (c *LicenseAssignmentsGetCall) Do() (*LicenseAssignment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{skuId}", url.QueryEscape(c.skuId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(LicenseAssignment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get license assignment of a particular product and sku for a user",
	//   "httpMethod": "GET",
	//   "id": "licensing.licenseAssignments.get",
	//   "parameterOrder": [
	//     "productId",
	//     "skuId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "productId": {
	//       "description": "Name for product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "skuId": {
	//       "description": "Name for sku",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "email id or unique Id of the user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{productId}/sku/{skuId}/user/{userId}",
	//   "response": {
	//     "$ref": "LicenseAssignment"
	//   }
	// }

}

// method id "licensing.licenseAssignments.insert":

type LicenseAssignmentsInsertCall struct {
	s                       *Service
	productId               string
	skuId                   string
	licenseassignmentinsert *LicenseAssignmentInsert
	opt_                    map[string]interface{}
}

// Insert: Assign License.
func (r *LicenseAssignmentsService) Insert(productId string, skuId string, licenseassignmentinsert *LicenseAssignmentInsert) *LicenseAssignmentsInsertCall {
	c := &LicenseAssignmentsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.productId = productId
	c.skuId = skuId
	c.licenseassignmentinsert = licenseassignmentinsert
	return c
}

func (c *LicenseAssignmentsInsertCall) Do() (*LicenseAssignment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.licenseassignmentinsert)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{skuId}", url.QueryEscape(c.skuId), 1)
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
	ret := new(LicenseAssignment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Assign License.",
	//   "httpMethod": "POST",
	//   "id": "licensing.licenseAssignments.insert",
	//   "parameterOrder": [
	//     "productId",
	//     "skuId"
	//   ],
	//   "parameters": {
	//     "productId": {
	//       "description": "Name for product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "skuId": {
	//       "description": "Name for sku",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{productId}/sku/{skuId}/user",
	//   "request": {
	//     "$ref": "LicenseAssignmentInsert"
	//   },
	//   "response": {
	//     "$ref": "LicenseAssignment"
	//   }
	// }

}

// method id "licensing.licenseAssignments.listForProduct":

type LicenseAssignmentsListForProductCall struct {
	s          *Service
	productId  string
	customerId string
	opt_       map[string]interface{}
}

// ListForProduct: List license assignments for given product of the
// customer.
func (r *LicenseAssignmentsService) ListForProduct(productId string, customerId string) *LicenseAssignmentsListForProductCall {
	c := &LicenseAssignmentsListForProductCall{s: r.s, opt_: make(map[string]interface{})}
	c.productId = productId
	c.customerId = customerId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of campaigns to return at one time. Must be positive.  Default value
// is 100.
func (c *LicenseAssignmentsListForProductCall) MaxResults(maxResults int64) *LicenseAssignmentsListForProductCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Token to fetch the
// next page. By default server will return first page
func (c *LicenseAssignmentsListForProductCall) PageToken(pageToken string) *LicenseAssignmentsListForProductCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *LicenseAssignmentsListForProductCall) Do() (*LicenseAssignmentList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("customerId", fmt.Sprintf("%v", c.customerId))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/users")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
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
	ret := new(LicenseAssignmentList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List license assignments for given product of the customer.",
	//   "httpMethod": "GET",
	//   "id": "licensing.licenseAssignments.listForProduct",
	//   "parameterOrder": [
	//     "productId",
	//     "customerId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "CustomerId represents the customer for whom licenseassignments are queried",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Maximum number of campaigns to return at one time. Must be positive. Optional. Default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "default": "",
	//       "description": "Token to fetch the next page.Optional. By default server will return first page",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "Name for product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{productId}/users",
	//   "response": {
	//     "$ref": "LicenseAssignmentList"
	//   }
	// }

}

// method id "licensing.licenseAssignments.listForProductAndSku":

type LicenseAssignmentsListForProductAndSkuCall struct {
	s          *Service
	productId  string
	skuId      string
	customerId string
	opt_       map[string]interface{}
}

// ListForProductAndSku: List license assignments for given product and
// sku of the customer.
func (r *LicenseAssignmentsService) ListForProductAndSku(productId string, skuId string, customerId string) *LicenseAssignmentsListForProductAndSkuCall {
	c := &LicenseAssignmentsListForProductAndSkuCall{s: r.s, opt_: make(map[string]interface{})}
	c.productId = productId
	c.skuId = skuId
	c.customerId = customerId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of campaigns to return at one time. Must be positive.  Default value
// is 100.
func (c *LicenseAssignmentsListForProductAndSkuCall) MaxResults(maxResults int64) *LicenseAssignmentsListForProductAndSkuCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Token to fetch the
// next page. By default server will return first page
func (c *LicenseAssignmentsListForProductAndSkuCall) PageToken(pageToken string) *LicenseAssignmentsListForProductAndSkuCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *LicenseAssignmentsListForProductAndSkuCall) Do() (*LicenseAssignmentList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("customerId", fmt.Sprintf("%v", c.customerId))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/users")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{skuId}", url.QueryEscape(c.skuId), 1)
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
	ret := new(LicenseAssignmentList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List license assignments for given product and sku of the customer.",
	//   "httpMethod": "GET",
	//   "id": "licensing.licenseAssignments.listForProductAndSku",
	//   "parameterOrder": [
	//     "productId",
	//     "skuId",
	//     "customerId"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "CustomerId represents the customer for whom licenseassignments are queried",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Maximum number of campaigns to return at one time. Must be positive. Optional. Default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "default": "",
	//       "description": "Token to fetch the next page.Optional. By default server will return first page",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "Name for product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "skuId": {
	//       "description": "Name for sku",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{productId}/sku/{skuId}/users",
	//   "response": {
	//     "$ref": "LicenseAssignmentList"
	//   }
	// }

}

// method id "licensing.licenseAssignments.patch":

type LicenseAssignmentsPatchCall struct {
	s                 *Service
	productId         string
	skuId             string
	userId            string
	licenseassignment *LicenseAssignment
	opt_              map[string]interface{}
}

// Patch: Assign License. This method supports patch semantics.
func (r *LicenseAssignmentsService) Patch(productId string, skuId string, userId string, licenseassignment *LicenseAssignment) *LicenseAssignmentsPatchCall {
	c := &LicenseAssignmentsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	c.licenseassignment = licenseassignment
	return c
}

func (c *LicenseAssignmentsPatchCall) Do() (*LicenseAssignment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.licenseassignment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{skuId}", url.QueryEscape(c.skuId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(LicenseAssignment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Assign License. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "licensing.licenseAssignments.patch",
	//   "parameterOrder": [
	//     "productId",
	//     "skuId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "productId": {
	//       "description": "Name for product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "skuId": {
	//       "description": "Name for sku for which license would be revoked",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "email id or unique Id of the user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{productId}/sku/{skuId}/user/{userId}",
	//   "request": {
	//     "$ref": "LicenseAssignment"
	//   },
	//   "response": {
	//     "$ref": "LicenseAssignment"
	//   }
	// }

}

// method id "licensing.licenseAssignments.update":

type LicenseAssignmentsUpdateCall struct {
	s                 *Service
	productId         string
	skuId             string
	userId            string
	licenseassignment *LicenseAssignment
	opt_              map[string]interface{}
}

// Update: Assign License.
func (r *LicenseAssignmentsService) Update(productId string, skuId string, userId string, licenseassignment *LicenseAssignment) *LicenseAssignmentsUpdateCall {
	c := &LicenseAssignmentsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	c.licenseassignment = licenseassignment
	return c
}

func (c *LicenseAssignmentsUpdateCall) Do() (*LicenseAssignment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.licenseassignment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{skuId}", url.QueryEscape(c.skuId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(LicenseAssignment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Assign License.",
	//   "httpMethod": "PUT",
	//   "id": "licensing.licenseAssignments.update",
	//   "parameterOrder": [
	//     "productId",
	//     "skuId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "productId": {
	//       "description": "Name for product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "skuId": {
	//       "description": "Name for sku for which license would be revoked",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "email id or unique Id of the user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{productId}/sku/{skuId}/user/{userId}",
	//   "request": {
	//     "$ref": "LicenseAssignment"
	//   },
	//   "response": {
	//     "$ref": "LicenseAssignment"
	//   }
	// }

}
