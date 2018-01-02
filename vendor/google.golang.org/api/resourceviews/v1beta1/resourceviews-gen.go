// Package resourceviews provides access to the Resource Views API.
//
// See https://developers.google.com/compute/
//
// Usage example:
//
//   import "google.golang.org/api/resourceviews/v1beta1"
//   ...
//   resourceviewsService, err := resourceviews.New(oauthHttpClient)
package resourceviews // import "google.golang.org/api/resourceviews/v1beta1"

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

const apiId = "resourceviews:v1beta1"
const apiName = "resourceviews"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/resourceviews/v1beta1/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// View and manage your Google Compute Engine resources
	ComputeScope = "https://www.googleapis.com/auth/compute"

	// View your Google Compute Engine resources
	ComputeReadonlyScope = "https://www.googleapis.com/auth/compute.readonly"

	// View and manage your Google Cloud Platform management resources and
	// deployment status information
	NdevCloudmanScope = "https://www.googleapis.com/auth/ndev.cloudman"

	// View your Google Cloud Platform management resources and deployment
	// status information
	NdevCloudmanReadonlyScope = "https://www.googleapis.com/auth/ndev.cloudman.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.RegionViews = NewRegionViewsService(s)
	s.ZoneViews = NewZoneViewsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	RegionViews *RegionViewsService

	ZoneViews *ZoneViewsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewRegionViewsService(s *Service) *RegionViewsService {
	rs := &RegionViewsService{s: s}
	return rs
}

type RegionViewsService struct {
	s *Service
}

func NewZoneViewsService(s *Service) *ZoneViewsService {
	rs := &ZoneViewsService{s: s}
	return rs
}

type ZoneViewsService struct {
	s *Service
}

// Label: The Label to be applied to the resource views.
type Label struct {
	// Key: Key of the label.
	Key string `json:"key,omitempty"`

	// Value: Value of the label.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Key") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Label) MarshalJSON() ([]byte, error) {
	type noMethod Label
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RegionViewsAddResourcesRequest: The request to add resources to the
// resource view.
type RegionViewsAddResourcesRequest struct {
	// Resources: The list of resources to be added.
	Resources []string `json:"resources,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Resources") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resources") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RegionViewsAddResourcesRequest) MarshalJSON() ([]byte, error) {
	type noMethod RegionViewsAddResourcesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RegionViewsInsertResponse: The response to a resource view insert
// request.
type RegionViewsInsertResponse struct {
	// Resource: The resource view object inserted.
	Resource *ResourceView `json:"resource,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Resource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resource") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RegionViewsInsertResponse) MarshalJSON() ([]byte, error) {
	type noMethod RegionViewsInsertResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RegionViewsListResourcesResponse: The response to the list resource
// request.
type RegionViewsListResourcesResponse struct {
	// Members: The resources in the view.
	Members []string `json:"members,omitempty"`

	// NextPageToken: A token used for pagination.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Members") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Members") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RegionViewsListResourcesResponse) MarshalJSON() ([]byte, error) {
	type noMethod RegionViewsListResourcesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RegionViewsListResponse: The response to the list resource view
// request.
type RegionViewsListResponse struct {
	// NextPageToken: A token used for pagination.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ResourceViews: The list of resource views that meet the criteria.
	ResourceViews []*ResourceView `json:"resourceViews,omitempty"`

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

func (s *RegionViewsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod RegionViewsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RegionViewsRemoveResourcesRequest: The request to remove resources
// from the resource view.
type RegionViewsRemoveResourcesRequest struct {
	// Resources: The list of resources to be removed.
	Resources []string `json:"resources,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Resources") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resources") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RegionViewsRemoveResourcesRequest) MarshalJSON() ([]byte, error) {
	type noMethod RegionViewsRemoveResourcesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ResourceView: The resource view object.
type ResourceView struct {
	// CreationTime: The creation time of the resource view.
	CreationTime string `json:"creationTime,omitempty"`

	// Description: The detailed description of the resource view.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] The ID of the resource view.
	Id string `json:"id,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Labels: The labels for events.
	Labels []*Label `json:"labels,omitempty"`

	// LastModified: The last modified time of the view. Not supported yet.
	LastModified string `json:"lastModified,omitempty"`

	// Members: A list of all resources in the resource view.
	Members []string `json:"members,omitempty"`

	// Name: The name of the resource view.
	Name string `json:"name,omitempty"`

	// NumMembers: The total number of resources in the resource view.
	NumMembers int64 `json:"numMembers,omitempty"`

	// SelfLink: [Output Only] A self-link to the resource view.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ResourceView) MarshalJSON() ([]byte, error) {
	type noMethod ResourceView
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ZoneViewsAddResourcesRequest: The request to add resources to the
// resource view.
type ZoneViewsAddResourcesRequest struct {
	// Resources: The list of resources to be added.
	Resources []string `json:"resources,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Resources") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resources") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ZoneViewsAddResourcesRequest) MarshalJSON() ([]byte, error) {
	type noMethod ZoneViewsAddResourcesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ZoneViewsInsertResponse: The response to an insert request.
type ZoneViewsInsertResponse struct {
	// Resource: The resource view object that has been inserted.
	Resource *ResourceView `json:"resource,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Resource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resource") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ZoneViewsInsertResponse) MarshalJSON() ([]byte, error) {
	type noMethod ZoneViewsInsertResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ZoneViewsListResourcesResponse: The response to a list resource
// request.
type ZoneViewsListResourcesResponse struct {
	// Members: The full URL of resources in the view.
	Members []string `json:"members,omitempty"`

	// NextPageToken: A token used for pagination.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Members") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Members") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ZoneViewsListResourcesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ZoneViewsListResourcesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ZoneViewsListResponse: The response to a list request.
type ZoneViewsListResponse struct {
	// NextPageToken: A token used for pagination.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ResourceViews: The result that contains all resource views that meet
	// the criteria.
	ResourceViews []*ResourceView `json:"resourceViews,omitempty"`

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

func (s *ZoneViewsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ZoneViewsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ZoneViewsRemoveResourcesRequest: The request to remove resources from
// the resource view.
type ZoneViewsRemoveResourcesRequest struct {
	// Resources: The list of resources to be removed.
	Resources []string `json:"resources,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Resources") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resources") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ZoneViewsRemoveResourcesRequest) MarshalJSON() ([]byte, error) {
	type noMethod ZoneViewsRemoveResourcesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "resourceviews.regionViews.addresources":

type RegionViewsAddresourcesCall struct {
	s                              *Service
	projectName                    string
	region                         string
	resourceViewName               string
	regionviewsaddresourcesrequest *RegionViewsAddResourcesRequest
	urlParams_                     gensupport.URLParams
	ctx_                           context.Context
	header_                        http.Header
}

// Addresources: Add resources to the view.
func (r *RegionViewsService) Addresources(projectName string, region string, resourceViewName string, regionviewsaddresourcesrequest *RegionViewsAddResourcesRequest) *RegionViewsAddresourcesCall {
	c := &RegionViewsAddresourcesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.region = region
	c.resourceViewName = resourceViewName
	c.regionviewsaddresourcesrequest = regionviewsaddresourcesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionViewsAddresourcesCall) Fields(s ...googleapi.Field) *RegionViewsAddresourcesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RegionViewsAddresourcesCall) Context(ctx context.Context) *RegionViewsAddresourcesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RegionViewsAddresourcesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RegionViewsAddresourcesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.regionviewsaddresourcesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/regions/{region}/resourceViews/{resourceViewName}/addResources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"region":           c.region,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.regionViews.addresources" call.
func (c *RegionViewsAddresourcesCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Add resources to the view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.regionViews.addresources",
	//   "parameterOrder": [
	//     "projectName",
	//     "region",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The region name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/regions/{region}/resourceViews/{resourceViewName}/addResources",
	//   "request": {
	//     "$ref": "RegionViewsAddResourcesRequest"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "resourceviews.regionViews.delete":

type RegionViewsDeleteCall struct {
	s                *Service
	projectName      string
	region           string
	resourceViewName string
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Delete: Delete a resource view.
func (r *RegionViewsService) Delete(projectName string, region string, resourceViewName string) *RegionViewsDeleteCall {
	c := &RegionViewsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.region = region
	c.resourceViewName = resourceViewName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionViewsDeleteCall) Fields(s ...googleapi.Field) *RegionViewsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RegionViewsDeleteCall) Context(ctx context.Context) *RegionViewsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RegionViewsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RegionViewsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/regions/{region}/resourceViews/{resourceViewName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"region":           c.region,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.regionViews.delete" call.
func (c *RegionViewsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Delete a resource view.",
	//   "httpMethod": "DELETE",
	//   "id": "resourceviews.regionViews.delete",
	//   "parameterOrder": [
	//     "projectName",
	//     "region",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The region name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/regions/{region}/resourceViews/{resourceViewName}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "resourceviews.regionViews.get":

type RegionViewsGetCall struct {
	s                *Service
	projectName      string
	region           string
	resourceViewName string
	urlParams_       gensupport.URLParams
	ifNoneMatch_     string
	ctx_             context.Context
	header_          http.Header
}

// Get: Get the information of a resource view.
func (r *RegionViewsService) Get(projectName string, region string, resourceViewName string) *RegionViewsGetCall {
	c := &RegionViewsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.region = region
	c.resourceViewName = resourceViewName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionViewsGetCall) Fields(s ...googleapi.Field) *RegionViewsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RegionViewsGetCall) IfNoneMatch(entityTag string) *RegionViewsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RegionViewsGetCall) Context(ctx context.Context) *RegionViewsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RegionViewsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RegionViewsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/regions/{region}/resourceViews/{resourceViewName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"region":           c.region,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.regionViews.get" call.
// Exactly one of *ResourceView or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ResourceView.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *RegionViewsGetCall) Do(opts ...googleapi.CallOption) (*ResourceView, error) {
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
	ret := &ResourceView{
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
	//   "description": "Get the information of a resource view.",
	//   "httpMethod": "GET",
	//   "id": "resourceviews.regionViews.get",
	//   "parameterOrder": [
	//     "projectName",
	//     "region",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The region name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/regions/{region}/resourceViews/{resourceViewName}",
	//   "response": {
	//     "$ref": "ResourceView"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "resourceviews.regionViews.insert":

type RegionViewsInsertCall struct {
	s            *Service
	projectName  string
	region       string
	resourceview *ResourceView
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Insert: Create a resource view.
func (r *RegionViewsService) Insert(projectName string, region string, resourceview *ResourceView) *RegionViewsInsertCall {
	c := &RegionViewsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.region = region
	c.resourceview = resourceview
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionViewsInsertCall) Fields(s ...googleapi.Field) *RegionViewsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RegionViewsInsertCall) Context(ctx context.Context) *RegionViewsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RegionViewsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RegionViewsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.resourceview)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/regions/{region}/resourceViews")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName": c.projectName,
		"region":      c.region,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.regionViews.insert" call.
// Exactly one of *RegionViewsInsertResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *RegionViewsInsertResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RegionViewsInsertCall) Do(opts ...googleapi.CallOption) (*RegionViewsInsertResponse, error) {
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
	ret := &RegionViewsInsertResponse{
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
	//   "description": "Create a resource view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.regionViews.insert",
	//   "parameterOrder": [
	//     "projectName",
	//     "region"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The region name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/regions/{region}/resourceViews",
	//   "request": {
	//     "$ref": "ResourceView"
	//   },
	//   "response": {
	//     "$ref": "RegionViewsInsertResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "resourceviews.regionViews.list":

type RegionViewsListCall struct {
	s            *Service
	projectName  string
	region       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List resource views.
func (r *RegionViewsService) List(projectName string, region string) *RegionViewsListCall {
	c := &RegionViewsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.region = region
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Acceptable values are 0 to 5000, inclusive.
// (Default: 5000)
func (c *RegionViewsListCall) MaxResults(maxResults int64) *RegionViewsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a
// nextPageToken returned by a previous list request. This token can be
// used to request the next page of results from a previous list
// request.
func (c *RegionViewsListCall) PageToken(pageToken string) *RegionViewsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionViewsListCall) Fields(s ...googleapi.Field) *RegionViewsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RegionViewsListCall) IfNoneMatch(entityTag string) *RegionViewsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RegionViewsListCall) Context(ctx context.Context) *RegionViewsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RegionViewsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RegionViewsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/regions/{region}/resourceViews")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName": c.projectName,
		"region":      c.region,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.regionViews.list" call.
// Exactly one of *RegionViewsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RegionViewsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RegionViewsListCall) Do(opts ...googleapi.CallOption) (*RegionViewsListResponse, error) {
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
	ret := &RegionViewsListResponse{
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
	//   "description": "List resource views.",
	//   "httpMethod": "GET",
	//   "id": "resourceviews.regionViews.list",
	//   "parameterOrder": [
	//     "projectName",
	//     "region"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "5000",
	//       "description": "Maximum count of results to be returned. Acceptable values are 0 to 5000, inclusive. (Default: 5000)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a nextPageToken returned by a previous list request. This token can be used to request the next page of results from a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The region name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/regions/{region}/resourceViews",
	//   "response": {
	//     "$ref": "RegionViewsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *RegionViewsListCall) Pages(ctx context.Context, f func(*RegionViewsListResponse) error) error {
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

// method id "resourceviews.regionViews.listresources":

type RegionViewsListresourcesCall struct {
	s                *Service
	projectName      string
	region           string
	resourceViewName string
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Listresources: List the resources in the view.
func (r *RegionViewsService) Listresources(projectName string, region string, resourceViewName string) *RegionViewsListresourcesCall {
	c := &RegionViewsListresourcesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.region = region
	c.resourceViewName = resourceViewName
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Acceptable values are 0 to 5000, inclusive.
// (Default: 5000)
func (c *RegionViewsListresourcesCall) MaxResults(maxResults int64) *RegionViewsListresourcesCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a
// nextPageToken returned by a previous list request. This token can be
// used to request the next page of results from a previous list
// request.
func (c *RegionViewsListresourcesCall) PageToken(pageToken string) *RegionViewsListresourcesCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionViewsListresourcesCall) Fields(s ...googleapi.Field) *RegionViewsListresourcesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RegionViewsListresourcesCall) Context(ctx context.Context) *RegionViewsListresourcesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RegionViewsListresourcesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RegionViewsListresourcesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/regions/{region}/resourceViews/{resourceViewName}/resources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"region":           c.region,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.regionViews.listresources" call.
// Exactly one of *RegionViewsListResourcesResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *RegionViewsListResourcesResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RegionViewsListresourcesCall) Do(opts ...googleapi.CallOption) (*RegionViewsListResourcesResponse, error) {
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
	ret := &RegionViewsListResourcesResponse{
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
	//   "description": "List the resources in the view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.regionViews.listresources",
	//   "parameterOrder": [
	//     "projectName",
	//     "region",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "5000",
	//       "description": "Maximum count of results to be returned. Acceptable values are 0 to 5000, inclusive. (Default: 5000)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a nextPageToken returned by a previous list request. This token can be used to request the next page of results from a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The region name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/regions/{region}/resourceViews/{resourceViewName}/resources",
	//   "response": {
	//     "$ref": "RegionViewsListResourcesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *RegionViewsListresourcesCall) Pages(ctx context.Context, f func(*RegionViewsListResourcesResponse) error) error {
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

// method id "resourceviews.regionViews.removeresources":

type RegionViewsRemoveresourcesCall struct {
	s                                 *Service
	projectName                       string
	region                            string
	resourceViewName                  string
	regionviewsremoveresourcesrequest *RegionViewsRemoveResourcesRequest
	urlParams_                        gensupport.URLParams
	ctx_                              context.Context
	header_                           http.Header
}

// Removeresources: Remove resources from the view.
func (r *RegionViewsService) Removeresources(projectName string, region string, resourceViewName string, regionviewsremoveresourcesrequest *RegionViewsRemoveResourcesRequest) *RegionViewsRemoveresourcesCall {
	c := &RegionViewsRemoveresourcesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.region = region
	c.resourceViewName = resourceViewName
	c.regionviewsremoveresourcesrequest = regionviewsremoveresourcesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionViewsRemoveresourcesCall) Fields(s ...googleapi.Field) *RegionViewsRemoveresourcesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RegionViewsRemoveresourcesCall) Context(ctx context.Context) *RegionViewsRemoveresourcesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RegionViewsRemoveresourcesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RegionViewsRemoveresourcesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.regionviewsremoveresourcesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/regions/{region}/resourceViews/{resourceViewName}/removeResources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"region":           c.region,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.regionViews.removeresources" call.
func (c *RegionViewsRemoveresourcesCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Remove resources from the view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.regionViews.removeresources",
	//   "parameterOrder": [
	//     "projectName",
	//     "region",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The region name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/regions/{region}/resourceViews/{resourceViewName}/removeResources",
	//   "request": {
	//     "$ref": "RegionViewsRemoveResourcesRequest"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "resourceviews.zoneViews.addresources":

type ZoneViewsAddresourcesCall struct {
	s                            *Service
	projectName                  string
	zone                         string
	resourceViewName             string
	zoneviewsaddresourcesrequest *ZoneViewsAddResourcesRequest
	urlParams_                   gensupport.URLParams
	ctx_                         context.Context
	header_                      http.Header
}

// Addresources: Add resources to the view.
func (r *ZoneViewsService) Addresources(projectName string, zone string, resourceViewName string, zoneviewsaddresourcesrequest *ZoneViewsAddResourcesRequest) *ZoneViewsAddresourcesCall {
	c := &ZoneViewsAddresourcesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.zone = zone
	c.resourceViewName = resourceViewName
	c.zoneviewsaddresourcesrequest = zoneviewsaddresourcesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneViewsAddresourcesCall) Fields(s ...googleapi.Field) *ZoneViewsAddresourcesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneViewsAddresourcesCall) Context(ctx context.Context) *ZoneViewsAddresourcesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneViewsAddresourcesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneViewsAddresourcesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.zoneviewsaddresourcesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/zones/{zone}/resourceViews/{resourceViewName}/addResources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"zone":             c.zone,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.zoneViews.addresources" call.
func (c *ZoneViewsAddresourcesCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Add resources to the view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.zoneViews.addresources",
	//   "parameterOrder": [
	//     "projectName",
	//     "zone",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The zone name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/zones/{zone}/resourceViews/{resourceViewName}/addResources",
	//   "request": {
	//     "$ref": "ZoneViewsAddResourcesRequest"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "resourceviews.zoneViews.delete":

type ZoneViewsDeleteCall struct {
	s                *Service
	projectName      string
	zone             string
	resourceViewName string
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Delete: Delete a resource view.
func (r *ZoneViewsService) Delete(projectName string, zone string, resourceViewName string) *ZoneViewsDeleteCall {
	c := &ZoneViewsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.zone = zone
	c.resourceViewName = resourceViewName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneViewsDeleteCall) Fields(s ...googleapi.Field) *ZoneViewsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneViewsDeleteCall) Context(ctx context.Context) *ZoneViewsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneViewsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneViewsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/zones/{zone}/resourceViews/{resourceViewName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"zone":             c.zone,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.zoneViews.delete" call.
func (c *ZoneViewsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Delete a resource view.",
	//   "httpMethod": "DELETE",
	//   "id": "resourceviews.zoneViews.delete",
	//   "parameterOrder": [
	//     "projectName",
	//     "zone",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The zone name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/zones/{zone}/resourceViews/{resourceViewName}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "resourceviews.zoneViews.get":

type ZoneViewsGetCall struct {
	s                *Service
	projectName      string
	zone             string
	resourceViewName string
	urlParams_       gensupport.URLParams
	ifNoneMatch_     string
	ctx_             context.Context
	header_          http.Header
}

// Get: Get the information of a zonal resource view.
func (r *ZoneViewsService) Get(projectName string, zone string, resourceViewName string) *ZoneViewsGetCall {
	c := &ZoneViewsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.zone = zone
	c.resourceViewName = resourceViewName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneViewsGetCall) Fields(s ...googleapi.Field) *ZoneViewsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ZoneViewsGetCall) IfNoneMatch(entityTag string) *ZoneViewsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneViewsGetCall) Context(ctx context.Context) *ZoneViewsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneViewsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneViewsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/zones/{zone}/resourceViews/{resourceViewName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"zone":             c.zone,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.zoneViews.get" call.
// Exactly one of *ResourceView or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ResourceView.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ZoneViewsGetCall) Do(opts ...googleapi.CallOption) (*ResourceView, error) {
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
	ret := &ResourceView{
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
	//   "description": "Get the information of a zonal resource view.",
	//   "httpMethod": "GET",
	//   "id": "resourceviews.zoneViews.get",
	//   "parameterOrder": [
	//     "projectName",
	//     "zone",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The zone name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/zones/{zone}/resourceViews/{resourceViewName}",
	//   "response": {
	//     "$ref": "ResourceView"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "resourceviews.zoneViews.insert":

type ZoneViewsInsertCall struct {
	s            *Service
	projectName  string
	zone         string
	resourceview *ResourceView
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Insert: Create a resource view.
func (r *ZoneViewsService) Insert(projectName string, zone string, resourceview *ResourceView) *ZoneViewsInsertCall {
	c := &ZoneViewsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.zone = zone
	c.resourceview = resourceview
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneViewsInsertCall) Fields(s ...googleapi.Field) *ZoneViewsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneViewsInsertCall) Context(ctx context.Context) *ZoneViewsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneViewsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneViewsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.resourceview)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/zones/{zone}/resourceViews")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName": c.projectName,
		"zone":        c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.zoneViews.insert" call.
// Exactly one of *ZoneViewsInsertResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ZoneViewsInsertResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ZoneViewsInsertCall) Do(opts ...googleapi.CallOption) (*ZoneViewsInsertResponse, error) {
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
	ret := &ZoneViewsInsertResponse{
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
	//   "description": "Create a resource view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.zoneViews.insert",
	//   "parameterOrder": [
	//     "projectName",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The zone name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/zones/{zone}/resourceViews",
	//   "request": {
	//     "$ref": "ResourceView"
	//   },
	//   "response": {
	//     "$ref": "ZoneViewsInsertResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "resourceviews.zoneViews.list":

type ZoneViewsListCall struct {
	s            *Service
	projectName  string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List resource views.
func (r *ZoneViewsService) List(projectName string, zone string) *ZoneViewsListCall {
	c := &ZoneViewsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.zone = zone
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Acceptable values are 0 to 5000, inclusive.
// (Default: 5000)
func (c *ZoneViewsListCall) MaxResults(maxResults int64) *ZoneViewsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a
// nextPageToken returned by a previous list request. This token can be
// used to request the next page of results from a previous list
// request.
func (c *ZoneViewsListCall) PageToken(pageToken string) *ZoneViewsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneViewsListCall) Fields(s ...googleapi.Field) *ZoneViewsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ZoneViewsListCall) IfNoneMatch(entityTag string) *ZoneViewsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneViewsListCall) Context(ctx context.Context) *ZoneViewsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneViewsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneViewsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/zones/{zone}/resourceViews")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName": c.projectName,
		"zone":        c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.zoneViews.list" call.
// Exactly one of *ZoneViewsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ZoneViewsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ZoneViewsListCall) Do(opts ...googleapi.CallOption) (*ZoneViewsListResponse, error) {
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
	ret := &ZoneViewsListResponse{
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
	//   "description": "List resource views.",
	//   "httpMethod": "GET",
	//   "id": "resourceviews.zoneViews.list",
	//   "parameterOrder": [
	//     "projectName",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "5000",
	//       "description": "Maximum count of results to be returned. Acceptable values are 0 to 5000, inclusive. (Default: 5000)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a nextPageToken returned by a previous list request. This token can be used to request the next page of results from a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The zone name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/zones/{zone}/resourceViews",
	//   "response": {
	//     "$ref": "ZoneViewsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ZoneViewsListCall) Pages(ctx context.Context, f func(*ZoneViewsListResponse) error) error {
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

// method id "resourceviews.zoneViews.listresources":

type ZoneViewsListresourcesCall struct {
	s                *Service
	projectName      string
	zone             string
	resourceViewName string
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Listresources: List the resources of the resource view.
func (r *ZoneViewsService) Listresources(projectName string, zone string, resourceViewName string) *ZoneViewsListresourcesCall {
	c := &ZoneViewsListresourcesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.zone = zone
	c.resourceViewName = resourceViewName
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Acceptable values are 0 to 5000, inclusive.
// (Default: 5000)
func (c *ZoneViewsListresourcesCall) MaxResults(maxResults int64) *ZoneViewsListresourcesCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a
// nextPageToken returned by a previous list request. This token can be
// used to request the next page of results from a previous list
// request.
func (c *ZoneViewsListresourcesCall) PageToken(pageToken string) *ZoneViewsListresourcesCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneViewsListresourcesCall) Fields(s ...googleapi.Field) *ZoneViewsListresourcesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneViewsListresourcesCall) Context(ctx context.Context) *ZoneViewsListresourcesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneViewsListresourcesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneViewsListresourcesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/zones/{zone}/resourceViews/{resourceViewName}/resources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"zone":             c.zone,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.zoneViews.listresources" call.
// Exactly one of *ZoneViewsListResourcesResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ZoneViewsListResourcesResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ZoneViewsListresourcesCall) Do(opts ...googleapi.CallOption) (*ZoneViewsListResourcesResponse, error) {
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
	ret := &ZoneViewsListResourcesResponse{
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
	//   "description": "List the resources of the resource view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.zoneViews.listresources",
	//   "parameterOrder": [
	//     "projectName",
	//     "zone",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "5000",
	//       "description": "Maximum count of results to be returned. Acceptable values are 0 to 5000, inclusive. (Default: 5000)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a nextPageToken returned by a previous list request. This token can be used to request the next page of results from a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The zone name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/zones/{zone}/resourceViews/{resourceViewName}/resources",
	//   "response": {
	//     "$ref": "ZoneViewsListResourcesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ZoneViewsListresourcesCall) Pages(ctx context.Context, f func(*ZoneViewsListResourcesResponse) error) error {
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

// method id "resourceviews.zoneViews.removeresources":

type ZoneViewsRemoveresourcesCall struct {
	s                               *Service
	projectName                     string
	zone                            string
	resourceViewName                string
	zoneviewsremoveresourcesrequest *ZoneViewsRemoveResourcesRequest
	urlParams_                      gensupport.URLParams
	ctx_                            context.Context
	header_                         http.Header
}

// Removeresources: Remove resources from the view.
func (r *ZoneViewsService) Removeresources(projectName string, zone string, resourceViewName string, zoneviewsremoveresourcesrequest *ZoneViewsRemoveResourcesRequest) *ZoneViewsRemoveresourcesCall {
	c := &ZoneViewsRemoveresourcesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectName = projectName
	c.zone = zone
	c.resourceViewName = resourceViewName
	c.zoneviewsremoveresourcesrequest = zoneviewsremoveresourcesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneViewsRemoveresourcesCall) Fields(s ...googleapi.Field) *ZoneViewsRemoveresourcesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneViewsRemoveresourcesCall) Context(ctx context.Context) *ZoneViewsRemoveresourcesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneViewsRemoveresourcesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneViewsRemoveresourcesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.zoneviewsremoveresourcesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectName}/zones/{zone}/resourceViews/{resourceViewName}/removeResources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectName":      c.projectName,
		"zone":             c.zone,
		"resourceViewName": c.resourceViewName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "resourceviews.zoneViews.removeresources" call.
func (c *ZoneViewsRemoveresourcesCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Remove resources from the view.",
	//   "httpMethod": "POST",
	//   "id": "resourceviews.zoneViews.removeresources",
	//   "parameterOrder": [
	//     "projectName",
	//     "zone",
	//     "resourceViewName"
	//   ],
	//   "parameters": {
	//     "projectName": {
	//       "description": "The project name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resourceViewName": {
	//       "description": "The name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The zone name of the resource view.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectName}/zones/{zone}/resourceViews/{resourceViewName}/removeResources",
	//   "request": {
	//     "$ref": "ZoneViewsRemoveResourcesRequest"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}
