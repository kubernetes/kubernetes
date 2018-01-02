// Package licensing provides access to the Enterprise License Manager API.
//
// See https://developers.google.com/google-apps/licensing/
//
// Usage example:
//
//   import "google.golang.org/api/licensing/v1"
//   ...
//   licensingService, err := licensing.New(oauthHttpClient)
package licensing // import "google.golang.org/api/licensing/v1"

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

const apiId = "licensing:v1"
const apiName = "licensing"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/apps/licensing/v1/product/"

// OAuth2 scopes used by this API.
const (
	// View and manage G Suite licenses for your domain
	AppsLicensingScope = "https://www.googleapis.com/auth/apps.licensing"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.LicenseAssignments = NewLicenseAssignmentsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	LicenseAssignments *LicenseAssignmentsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewLicenseAssignmentsService(s *Service) *LicenseAssignmentsService {
	rs := &LicenseAssignmentsService{s: s}
	return rs
}

type LicenseAssignmentsService struct {
	s *Service
}

// LicenseAssignment: Template for LiscenseAssignment Resource
type LicenseAssignment struct {
	// Etags: ETag of the resource.
	Etags string `json:"etags,omitempty"`

	// Kind: Identifies the resource as a LicenseAssignment.
	Kind string `json:"kind,omitempty"`

	// ProductId: Id of the product.
	ProductId string `json:"productId,omitempty"`

	// ProductName: Display Name of the product.
	ProductName string `json:"productName,omitempty"`

	// SelfLink: Link to this page.
	SelfLink string `json:"selfLink,omitempty"`

	// SkuId: Id of the sku of the product.
	SkuId string `json:"skuId,omitempty"`

	// SkuName: Display Name of the sku of the product.
	SkuName string `json:"skuName,omitempty"`

	// UserId: Email id of the user.
	UserId string `json:"userId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Etags") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Etags") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LicenseAssignment) MarshalJSON() ([]byte, error) {
	type noMethod LicenseAssignment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LicenseAssignmentInsert: Template for LicenseAssignment Insert
// request
type LicenseAssignmentInsert struct {
	// UserId: Email id of the user
	UserId string `json:"userId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "UserId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "UserId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LicenseAssignmentInsert) MarshalJSON() ([]byte, error) {
	type noMethod LicenseAssignmentInsert
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LicenseAssignmentList: LicesnseAssignment List for a given
// product/sku for a customer.
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Etag") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Etag") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LicenseAssignmentList) MarshalJSON() ([]byte, error) {
	type noMethod LicenseAssignmentList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "licensing.licenseAssignments.delete":

type LicenseAssignmentsDeleteCall struct {
	s          *Service
	productId  string
	skuId      string
	userId     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Revoke License.
func (r *LicenseAssignmentsService) Delete(productId string, skuId string, userId string) *LicenseAssignmentsDeleteCall {
	c := &LicenseAssignmentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicenseAssignmentsDeleteCall) Fields(s ...googleapi.Field) *LicenseAssignmentsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LicenseAssignmentsDeleteCall) Context(ctx context.Context) *LicenseAssignmentsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LicenseAssignmentsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LicenseAssignmentsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"productId": c.productId,
		"skuId":     c.skuId,
		"userId":    c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "licensing.licenseAssignments.delete" call.
func (c *LicenseAssignmentsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "path": "{productId}/sku/{skuId}/user/{userId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.licensing"
	//   ]
	// }

}

// method id "licensing.licenseAssignments.get":

type LicenseAssignmentsGetCall struct {
	s            *Service
	productId    string
	skuId        string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get license assignment of a particular product and sku for a
// user
func (r *LicenseAssignmentsService) Get(productId string, skuId string, userId string) *LicenseAssignmentsGetCall {
	c := &LicenseAssignmentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicenseAssignmentsGetCall) Fields(s ...googleapi.Field) *LicenseAssignmentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *LicenseAssignmentsGetCall) IfNoneMatch(entityTag string) *LicenseAssignmentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LicenseAssignmentsGetCall) Context(ctx context.Context) *LicenseAssignmentsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LicenseAssignmentsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LicenseAssignmentsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"productId": c.productId,
		"skuId":     c.skuId,
		"userId":    c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "licensing.licenseAssignments.get" call.
// Exactly one of *LicenseAssignment or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LicenseAssignment.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *LicenseAssignmentsGetCall) Do(opts ...googleapi.CallOption) (*LicenseAssignment, error) {
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
	ret := &LicenseAssignment{
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
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.licensing"
	//   ]
	// }

}

// method id "licensing.licenseAssignments.insert":

type LicenseAssignmentsInsertCall struct {
	s                       *Service
	productId               string
	skuId                   string
	licenseassignmentinsert *LicenseAssignmentInsert
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
	header_                 http.Header
}

// Insert: Assign License.
func (r *LicenseAssignmentsService) Insert(productId string, skuId string, licenseassignmentinsert *LicenseAssignmentInsert) *LicenseAssignmentsInsertCall {
	c := &LicenseAssignmentsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productId = productId
	c.skuId = skuId
	c.licenseassignmentinsert = licenseassignmentinsert
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicenseAssignmentsInsertCall) Fields(s ...googleapi.Field) *LicenseAssignmentsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LicenseAssignmentsInsertCall) Context(ctx context.Context) *LicenseAssignmentsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LicenseAssignmentsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LicenseAssignmentsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.licenseassignmentinsert)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"productId": c.productId,
		"skuId":     c.skuId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "licensing.licenseAssignments.insert" call.
// Exactly one of *LicenseAssignment or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LicenseAssignment.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *LicenseAssignmentsInsertCall) Do(opts ...googleapi.CallOption) (*LicenseAssignment, error) {
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
	ret := &LicenseAssignment{
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
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.licensing"
	//   ]
	// }

}

// method id "licensing.licenseAssignments.listForProduct":

type LicenseAssignmentsListForProductCall struct {
	s            *Service
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// ListForProduct: List license assignments for given product of the
// customer.
func (r *LicenseAssignmentsService) ListForProduct(productId string, customerId string) *LicenseAssignmentsListForProductCall {
	c := &LicenseAssignmentsListForProductCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productId = productId
	c.urlParams_.Set("customerId", customerId)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of campaigns to return at one time. Must be positive.  Default value
// is 100.
func (c *LicenseAssignmentsListForProductCall) MaxResults(maxResults int64) *LicenseAssignmentsListForProductCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token to fetch the
// next page. By default server will return first page
func (c *LicenseAssignmentsListForProductCall) PageToken(pageToken string) *LicenseAssignmentsListForProductCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicenseAssignmentsListForProductCall) Fields(s ...googleapi.Field) *LicenseAssignmentsListForProductCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *LicenseAssignmentsListForProductCall) IfNoneMatch(entityTag string) *LicenseAssignmentsListForProductCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LicenseAssignmentsListForProductCall) Context(ctx context.Context) *LicenseAssignmentsListForProductCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LicenseAssignmentsListForProductCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LicenseAssignmentsListForProductCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"productId": c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "licensing.licenseAssignments.listForProduct" call.
// Exactly one of *LicenseAssignmentList or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LicenseAssignmentList.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *LicenseAssignmentsListForProductCall) Do(opts ...googleapi.CallOption) (*LicenseAssignmentList, error) {
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
	ret := &LicenseAssignmentList{
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
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.licensing"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *LicenseAssignmentsListForProductCall) Pages(ctx context.Context, f func(*LicenseAssignmentList) error) error {
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

// method id "licensing.licenseAssignments.listForProductAndSku":

type LicenseAssignmentsListForProductAndSkuCall struct {
	s            *Service
	productId    string
	skuId        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// ListForProductAndSku: List license assignments for given product and
// sku of the customer.
func (r *LicenseAssignmentsService) ListForProductAndSku(productId string, skuId string, customerId string) *LicenseAssignmentsListForProductAndSkuCall {
	c := &LicenseAssignmentsListForProductAndSkuCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productId = productId
	c.skuId = skuId
	c.urlParams_.Set("customerId", customerId)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of campaigns to return at one time. Must be positive.  Default value
// is 100.
func (c *LicenseAssignmentsListForProductAndSkuCall) MaxResults(maxResults int64) *LicenseAssignmentsListForProductAndSkuCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token to fetch the
// next page. By default server will return first page
func (c *LicenseAssignmentsListForProductAndSkuCall) PageToken(pageToken string) *LicenseAssignmentsListForProductAndSkuCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicenseAssignmentsListForProductAndSkuCall) Fields(s ...googleapi.Field) *LicenseAssignmentsListForProductAndSkuCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *LicenseAssignmentsListForProductAndSkuCall) IfNoneMatch(entityTag string) *LicenseAssignmentsListForProductAndSkuCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LicenseAssignmentsListForProductAndSkuCall) Context(ctx context.Context) *LicenseAssignmentsListForProductAndSkuCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LicenseAssignmentsListForProductAndSkuCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LicenseAssignmentsListForProductAndSkuCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"productId": c.productId,
		"skuId":     c.skuId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "licensing.licenseAssignments.listForProductAndSku" call.
// Exactly one of *LicenseAssignmentList or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LicenseAssignmentList.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *LicenseAssignmentsListForProductAndSkuCall) Do(opts ...googleapi.CallOption) (*LicenseAssignmentList, error) {
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
	ret := &LicenseAssignmentList{
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
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.licensing"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *LicenseAssignmentsListForProductAndSkuCall) Pages(ctx context.Context, f func(*LicenseAssignmentList) error) error {
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

// method id "licensing.licenseAssignments.patch":

type LicenseAssignmentsPatchCall struct {
	s                 *Service
	productId         string
	skuId             string
	userId            string
	licenseassignment *LicenseAssignment
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Patch: Assign License. This method supports patch semantics.
func (r *LicenseAssignmentsService) Patch(productId string, skuId string, userId string, licenseassignment *LicenseAssignment) *LicenseAssignmentsPatchCall {
	c := &LicenseAssignmentsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	c.licenseassignment = licenseassignment
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicenseAssignmentsPatchCall) Fields(s ...googleapi.Field) *LicenseAssignmentsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LicenseAssignmentsPatchCall) Context(ctx context.Context) *LicenseAssignmentsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LicenseAssignmentsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LicenseAssignmentsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.licenseassignment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"productId": c.productId,
		"skuId":     c.skuId,
		"userId":    c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "licensing.licenseAssignments.patch" call.
// Exactly one of *LicenseAssignment or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LicenseAssignment.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *LicenseAssignmentsPatchCall) Do(opts ...googleapi.CallOption) (*LicenseAssignment, error) {
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
	ret := &LicenseAssignment{
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
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.licensing"
	//   ]
	// }

}

// method id "licensing.licenseAssignments.update":

type LicenseAssignmentsUpdateCall struct {
	s                 *Service
	productId         string
	skuId             string
	userId            string
	licenseassignment *LicenseAssignment
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Update: Assign License.
func (r *LicenseAssignmentsService) Update(productId string, skuId string, userId string, licenseassignment *LicenseAssignment) *LicenseAssignmentsUpdateCall {
	c := &LicenseAssignmentsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.productId = productId
	c.skuId = skuId
	c.userId = userId
	c.licenseassignment = licenseassignment
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicenseAssignmentsUpdateCall) Fields(s ...googleapi.Field) *LicenseAssignmentsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LicenseAssignmentsUpdateCall) Context(ctx context.Context) *LicenseAssignmentsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LicenseAssignmentsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LicenseAssignmentsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.licenseassignment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{productId}/sku/{skuId}/user/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"productId": c.productId,
		"skuId":     c.skuId,
		"userId":    c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "licensing.licenseAssignments.update" call.
// Exactly one of *LicenseAssignment or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LicenseAssignment.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *LicenseAssignmentsUpdateCall) Do(opts ...googleapi.CallOption) (*LicenseAssignment, error) {
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
	ret := &LicenseAssignment{
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
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.licensing"
	//   ]
	// }

}
