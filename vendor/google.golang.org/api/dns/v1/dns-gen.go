// Package dns provides access to the Google Cloud DNS API.
//
// See https://developers.google.com/cloud-dns
//
// Usage example:
//
//   import "google.golang.org/api/dns/v1"
//   ...
//   dnsService, err := dns.New(oauthHttpClient)
package dns

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

const apiId = "dns:v1"
const apiName = "dns"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/dns/v1/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// View your DNS records hosted by Google Cloud DNS
	NdevClouddnsReadonlyScope = "https://www.googleapis.com/auth/ndev.clouddns.readonly"

	// View and manage your DNS records hosted by Google Cloud DNS
	NdevClouddnsReadwriteScope = "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Changes = NewChangesService(s)
	s.ManagedZones = NewManagedZonesService(s)
	s.Projects = NewProjectsService(s)
	s.ResourceRecordSets = NewResourceRecordSetsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Changes *ChangesService

	ManagedZones *ManagedZonesService

	Projects *ProjectsService

	ResourceRecordSets *ResourceRecordSetsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewChangesService(s *Service) *ChangesService {
	rs := &ChangesService{s: s}
	return rs
}

type ChangesService struct {
	s *Service
}

func NewManagedZonesService(s *Service) *ManagedZonesService {
	rs := &ManagedZonesService{s: s}
	return rs
}

type ManagedZonesService struct {
	s *Service
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	return rs
}

type ProjectsService struct {
	s *Service
}

func NewResourceRecordSetsService(s *Service) *ResourceRecordSetsService {
	rs := &ResourceRecordSetsService{s: s}
	return rs
}

type ResourceRecordSetsService struct {
	s *Service
}

// Change: An atomic update to a collection of ResourceRecordSets.
type Change struct {
	// Additions: Which ResourceRecordSets to add?
	Additions []*ResourceRecordSet `json:"additions,omitempty"`

	// Deletions: Which ResourceRecordSets to remove? Must match existing
	// data exactly.
	Deletions []*ResourceRecordSet `json:"deletions,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "dns#change".
	Kind string `json:"kind,omitempty"`

	// StartTime: The time that this operation was started by the server
	// (output only). This is in RFC3339 text format.
	StartTime string `json:"startTime,omitempty"`

	// Status: Status of the operation (output only).
	//
	// Possible values:
	//   "done"
	//   "pending"
	Status string `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Additions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Change) MarshalJSON() ([]byte, error) {
	type noMethod Change
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ChangesListResponse: The response to a request to enumerate Changes
// to a ResourceRecordSets collection.
type ChangesListResponse struct {
	// Changes: The requested changes.
	Changes []*Change `json:"changes,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The presence of this field indicates that there exist
	// more results following your last page of results in pagination order.
	// To fetch them, make another list request using this value as your
	// pagination token.
	//
	// In this way you can retrieve the complete contents of even very large
	// collections one page at a time. However, if the contents of the
	// collection change between the first and last paginated list request,
	// the set of all elements returned will be an inconsistent view of the
	// collection. There is no way to retrieve a "snapshot" of collections
	// larger than the maximum page size.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Changes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ChangesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ChangesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ManagedZone: A zone is a subtree of the DNS namespace under one
// administrative responsibility. A ManagedZone is a resource that
// represents a DNS zone hosted by the Cloud DNS service.
type ManagedZone struct {
	// CreationTime: The time that this resource was created on the server.
	// This is in RFC3339 text format. Output only.
	CreationTime string `json:"creationTime,omitempty"`

	// Description: A mutable string of at most 1024 characters associated
	// with this resource for the user's convenience. Has no effect on the
	// managed zone's function.
	Description string `json:"description,omitempty"`

	// DnsName: The DNS name of this managed zone, for instance
	// "example.com.".
	DnsName string `json:"dnsName,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only)
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "dns#managedZone".
	Kind string `json:"kind,omitempty"`

	// Name: User assigned name for this resource. Must be unique within the
	// project. The name must be 1-32 characters long, must begin with a
	// letter, end with a letter or digit, and only contain lowercase
	// letters, digits or dashes.
	Name string `json:"name,omitempty"`

	// NameServerSet: Optionally specifies the NameServerSet for this
	// ManagedZone. A NameServerSet is a set of DNS name servers that all
	// host the same ManagedZones. Most users will leave this field unset.
	NameServerSet string `json:"nameServerSet,omitempty"`

	// NameServers: Delegate your managed_zone to these virtual name
	// servers; defined by the server (output only)
	NameServers []string `json:"nameServers,omitempty"`

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
}

func (s *ManagedZone) MarshalJSON() ([]byte, error) {
	type noMethod ManagedZone
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ManagedZonesListResponse struct {
	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// ManagedZones: The managed zone resources.
	ManagedZones []*ManagedZone `json:"managedZones,omitempty"`

	// NextPageToken: The presence of this field indicates that there exist
	// more results following your last page of results in pagination order.
	// To fetch them, make another list request using this value as your
	// page token.
	//
	// In this way you can retrieve the complete contents of even very large
	// collections one page at a time. However, if the contents of the
	// collection change between the first and last paginated list request,
	// the set of all elements returned will be an inconsistent view of the
	// collection. There is no way to retrieve a consistent snapshot of a
	// collection larger than the maximum page size.
	NextPageToken string `json:"nextPageToken,omitempty"`

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

func (s *ManagedZonesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ManagedZonesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Project: A project resource. The project is a top level container for
// resources including Cloud DNS ManagedZones. Projects can be created
// only in the APIs console.
type Project struct {
	// Id: User assigned unique identifier for the resource (output only).
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "dns#project".
	Kind string `json:"kind,omitempty"`

	// Number: Unique numeric identifier for the resource; defined by the
	// server (output only).
	Number uint64 `json:"number,omitempty,string"`

	// Quota: Quotas assigned to this project (output only).
	Quota *Quota `json:"quota,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Project) MarshalJSON() ([]byte, error) {
	type noMethod Project
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Quota: Limits associated with a Project.
type Quota struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "dns#quota".
	Kind string `json:"kind,omitempty"`

	// ManagedZones: Maximum allowed number of managed zones in the project.
	ManagedZones int64 `json:"managedZones,omitempty"`

	// ResourceRecordsPerRrset: Maximum allowed number of ResourceRecords
	// per ResourceRecordSet.
	ResourceRecordsPerRrset int64 `json:"resourceRecordsPerRrset,omitempty"`

	// RrsetAdditionsPerChange: Maximum allowed number of ResourceRecordSets
	// to add per ChangesCreateRequest.
	RrsetAdditionsPerChange int64 `json:"rrsetAdditionsPerChange,omitempty"`

	// RrsetDeletionsPerChange: Maximum allowed number of ResourceRecordSets
	// to delete per ChangesCreateRequest.
	RrsetDeletionsPerChange int64 `json:"rrsetDeletionsPerChange,omitempty"`

	// RrsetsPerManagedZone: Maximum allowed number of ResourceRecordSets
	// per zone in the project.
	RrsetsPerManagedZone int64 `json:"rrsetsPerManagedZone,omitempty"`

	// TotalRrdataSizePerChange: Maximum allowed size for total rrdata in
	// one ChangesCreateRequest in bytes.
	TotalRrdataSizePerChange int64 `json:"totalRrdataSizePerChange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Quota) MarshalJSON() ([]byte, error) {
	type noMethod Quota
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ResourceRecordSet: A unit of data that will be returned by the DNS
// servers.
type ResourceRecordSet struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "dns#resourceRecordSet".
	Kind string `json:"kind,omitempty"`

	// Name: For example, www.example.com.
	Name string `json:"name,omitempty"`

	// Rrdatas: As defined in RFC 1035 (section 5) and RFC 1034 (section
	// 3.6.1).
	Rrdatas []string `json:"rrdatas,omitempty"`

	// Ttl: Number of seconds that this ResourceRecordSet can be cached by
	// resolvers.
	Ttl int64 `json:"ttl,omitempty"`

	// Type: The identifier of a supported record type, for example, A,
	// AAAA, MX, TXT, and so on.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ResourceRecordSet) MarshalJSON() ([]byte, error) {
	type noMethod ResourceRecordSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ResourceRecordSetsListResponse struct {
	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The presence of this field indicates that there exist
	// more results following your last page of results in pagination order.
	// To fetch them, make another list request using this value as your
	// pagination token.
	//
	// In this way you can retrieve the complete contents of even very large
	// collections one page at a time. However, if the contents of the
	// collection change between the first and last paginated list request,
	// the set of all elements returned will be an inconsistent view of the
	// collection. There is no way to retrieve a consistent snapshot of a
	// collection larger than the maximum page size.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Rrsets: The resource record set resources.
	Rrsets []*ResourceRecordSet `json:"rrsets,omitempty"`

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

func (s *ResourceRecordSetsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ResourceRecordSetsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "dns.changes.create":

type ChangesCreateCall struct {
	s           *Service
	project     string
	managedZone string
	change      *Change
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Create: Atomically update the ResourceRecordSet collection.
func (r *ChangesService) Create(project string, managedZone string, change *Change) *ChangesCreateCall {
	c := &ChangesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.managedZone = managedZone
	c.change = change
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChangesCreateCall) Fields(s ...googleapi.Field) *ChangesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChangesCreateCall) Context(ctx context.Context) *ChangesCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ChangesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.change)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/changes")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":     c.project,
		"managedZone": c.managedZone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.changes.create" call.
// Exactly one of *Change or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Change.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ChangesCreateCall) Do(opts ...googleapi.CallOption) (*Change, error) {
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
	ret := &Change{
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
	//   "description": "Atomically update the ResourceRecordSet collection.",
	//   "httpMethod": "POST",
	//   "id": "dns.changes.create",
	//   "parameterOrder": [
	//     "project",
	//     "managedZone"
	//   ],
	//   "parameters": {
	//     "managedZone": {
	//       "description": "Identifies the managed zone addressed by this request. Can be the managed zone name or id.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones/{managedZone}/changes",
	//   "request": {
	//     "$ref": "Change"
	//   },
	//   "response": {
	//     "$ref": "Change"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.changes.get":

type ChangesGetCall struct {
	s            *Service
	project      string
	managedZone  string
	changeId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Fetch the representation of an existing Change.
func (r *ChangesService) Get(project string, managedZone string, changeId string) *ChangesGetCall {
	c := &ChangesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.managedZone = managedZone
	c.changeId = changeId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChangesGetCall) Fields(s ...googleapi.Field) *ChangesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ChangesGetCall) IfNoneMatch(entityTag string) *ChangesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChangesGetCall) Context(ctx context.Context) *ChangesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ChangesGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/changes/{changeId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":     c.project,
		"managedZone": c.managedZone,
		"changeId":    c.changeId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.changes.get" call.
// Exactly one of *Change or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Change.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ChangesGetCall) Do(opts ...googleapi.CallOption) (*Change, error) {
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
	ret := &Change{
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
	//   "description": "Fetch the representation of an existing Change.",
	//   "httpMethod": "GET",
	//   "id": "dns.changes.get",
	//   "parameterOrder": [
	//     "project",
	//     "managedZone",
	//     "changeId"
	//   ],
	//   "parameters": {
	//     "changeId": {
	//       "description": "The identifier of the requested change, from a previous ResourceRecordSetsChangeResponse.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedZone": {
	//       "description": "Identifies the managed zone addressed by this request. Can be the managed zone name or id.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones/{managedZone}/changes/{changeId}",
	//   "response": {
	//     "$ref": "Change"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.changes.list":

type ChangesListCall struct {
	s            *Service
	project      string
	managedZone  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Enumerate Changes to a ResourceRecordSet collection.
func (r *ChangesService) List(project string, managedZone string) *ChangesListCall {
	c := &ChangesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.managedZone = managedZone
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to be returned. If unspecified, the server will decide how
// many results to return.
func (c *ChangesListCall) MaxResults(maxResults int64) *ChangesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": A tag returned by
// a previous list request that was truncated. Use this parameter to
// continue a previous list request.
func (c *ChangesListCall) PageToken(pageToken string) *ChangesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// SortBy sets the optional parameter "sortBy": Sorting criterion. The
// only supported value is change sequence.
//
// Possible values:
//   "changeSequence" (default)
func (c *ChangesListCall) SortBy(sortBy string) *ChangesListCall {
	c.urlParams_.Set("sortBy", sortBy)
	return c
}

// SortOrder sets the optional parameter "sortOrder": Sorting order
// direction: 'ascending' or 'descending'.
func (c *ChangesListCall) SortOrder(sortOrder string) *ChangesListCall {
	c.urlParams_.Set("sortOrder", sortOrder)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChangesListCall) Fields(s ...googleapi.Field) *ChangesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ChangesListCall) IfNoneMatch(entityTag string) *ChangesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChangesListCall) Context(ctx context.Context) *ChangesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ChangesListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/changes")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":     c.project,
		"managedZone": c.managedZone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.changes.list" call.
// Exactly one of *ChangesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ChangesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ChangesListCall) Do(opts ...googleapi.CallOption) (*ChangesListResponse, error) {
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
	ret := &ChangesListResponse{
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
	//   "description": "Enumerate Changes to a ResourceRecordSet collection.",
	//   "httpMethod": "GET",
	//   "id": "dns.changes.list",
	//   "parameterOrder": [
	//     "project",
	//     "managedZone"
	//   ],
	//   "parameters": {
	//     "managedZone": {
	//       "description": "Identifies the managed zone addressed by this request. Can be the managed zone name or id.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Optional. Maximum number of results to be returned. If unspecified, the server will decide how many results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. A tag returned by a previous list request that was truncated. Use this parameter to continue a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sortBy": {
	//       "default": "changeSequence",
	//       "description": "Sorting criterion. The only supported value is change sequence.",
	//       "enum": [
	//         "changeSequence"
	//       ],
	//       "enumDescriptions": [
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sortOrder": {
	//       "description": "Sorting order direction: 'ascending' or 'descending'.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones/{managedZone}/changes",
	//   "response": {
	//     "$ref": "ChangesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ChangesListCall) Pages(ctx context.Context, f func(*ChangesListResponse) error) error {
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

// method id "dns.managedZones.create":

type ManagedZonesCreateCall struct {
	s           *Service
	project     string
	managedzone *ManagedZone
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Create: Create a new ManagedZone.
func (r *ManagedZonesService) Create(project string, managedzone *ManagedZone) *ManagedZonesCreateCall {
	c := &ManagedZonesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.managedzone = managedzone
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedZonesCreateCall) Fields(s ...googleapi.Field) *ManagedZonesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedZonesCreateCall) Context(ctx context.Context) *ManagedZonesCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ManagedZonesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.managedzone)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.managedZones.create" call.
// Exactly one of *ManagedZone or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ManagedZone.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ManagedZonesCreateCall) Do(opts ...googleapi.CallOption) (*ManagedZone, error) {
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
	ret := &ManagedZone{
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
	//   "description": "Create a new ManagedZone.",
	//   "httpMethod": "POST",
	//   "id": "dns.managedZones.create",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones",
	//   "request": {
	//     "$ref": "ManagedZone"
	//   },
	//   "response": {
	//     "$ref": "ManagedZone"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.managedZones.delete":

type ManagedZonesDeleteCall struct {
	s           *Service
	project     string
	managedZone string
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Delete: Delete a previously created ManagedZone.
func (r *ManagedZonesService) Delete(project string, managedZone string) *ManagedZonesDeleteCall {
	c := &ManagedZonesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.managedZone = managedZone
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedZonesDeleteCall) Fields(s ...googleapi.Field) *ManagedZonesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedZonesDeleteCall) Context(ctx context.Context) *ManagedZonesDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ManagedZonesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":     c.project,
		"managedZone": c.managedZone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.managedZones.delete" call.
func (c *ManagedZonesDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Delete a previously created ManagedZone.",
	//   "httpMethod": "DELETE",
	//   "id": "dns.managedZones.delete",
	//   "parameterOrder": [
	//     "project",
	//     "managedZone"
	//   ],
	//   "parameters": {
	//     "managedZone": {
	//       "description": "Identifies the managed zone addressed by this request. Can be the managed zone name or id.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones/{managedZone}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.managedZones.get":

type ManagedZonesGetCall struct {
	s            *Service
	project      string
	managedZone  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Fetch the representation of an existing ManagedZone.
func (r *ManagedZonesService) Get(project string, managedZone string) *ManagedZonesGetCall {
	c := &ManagedZonesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.managedZone = managedZone
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedZonesGetCall) Fields(s ...googleapi.Field) *ManagedZonesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManagedZonesGetCall) IfNoneMatch(entityTag string) *ManagedZonesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedZonesGetCall) Context(ctx context.Context) *ManagedZonesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ManagedZonesGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":     c.project,
		"managedZone": c.managedZone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.managedZones.get" call.
// Exactly one of *ManagedZone or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ManagedZone.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ManagedZonesGetCall) Do(opts ...googleapi.CallOption) (*ManagedZone, error) {
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
	ret := &ManagedZone{
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
	//   "description": "Fetch the representation of an existing ManagedZone.",
	//   "httpMethod": "GET",
	//   "id": "dns.managedZones.get",
	//   "parameterOrder": [
	//     "project",
	//     "managedZone"
	//   ],
	//   "parameters": {
	//     "managedZone": {
	//       "description": "Identifies the managed zone addressed by this request. Can be the managed zone name or id.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones/{managedZone}",
	//   "response": {
	//     "$ref": "ManagedZone"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.managedZones.list":

type ManagedZonesListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Enumerate ManagedZones that have been created but not yet
// deleted.
func (r *ManagedZonesService) List(project string) *ManagedZonesListCall {
	c := &ManagedZonesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// DnsName sets the optional parameter "dnsName": Restricts the list to
// return only zones with this domain name.
func (c *ManagedZonesListCall) DnsName(dnsName string) *ManagedZonesListCall {
	c.urlParams_.Set("dnsName", dnsName)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to be returned. If unspecified, the server will decide how
// many results to return.
func (c *ManagedZonesListCall) MaxResults(maxResults int64) *ManagedZonesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": A tag returned by
// a previous list request that was truncated. Use this parameter to
// continue a previous list request.
func (c *ManagedZonesListCall) PageToken(pageToken string) *ManagedZonesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedZonesListCall) Fields(s ...googleapi.Field) *ManagedZonesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManagedZonesListCall) IfNoneMatch(entityTag string) *ManagedZonesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedZonesListCall) Context(ctx context.Context) *ManagedZonesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ManagedZonesListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.managedZones.list" call.
// Exactly one of *ManagedZonesListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ManagedZonesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManagedZonesListCall) Do(opts ...googleapi.CallOption) (*ManagedZonesListResponse, error) {
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
	ret := &ManagedZonesListResponse{
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
	//   "description": "Enumerate ManagedZones that have been created but not yet deleted.",
	//   "httpMethod": "GET",
	//   "id": "dns.managedZones.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "dnsName": {
	//       "description": "Restricts the list to return only zones with this domain name.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Optional. Maximum number of results to be returned. If unspecified, the server will decide how many results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. A tag returned by a previous list request that was truncated. Use this parameter to continue a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones",
	//   "response": {
	//     "$ref": "ManagedZonesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ManagedZonesListCall) Pages(ctx context.Context, f func(*ManagedZonesListResponse) error) error {
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

// method id "dns.projects.get":

type ProjectsGetCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Fetch the representation of an existing Project.
func (r *ProjectsService) Get(project string) *ProjectsGetCall {
	c := &ProjectsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsGetCall) Fields(s ...googleapi.Field) *ProjectsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsGetCall) IfNoneMatch(entityTag string) *ProjectsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsGetCall) Context(ctx context.Context) *ProjectsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.projects.get" call.
// Exactly one of *Project or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Project.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsGetCall) Do(opts ...googleapi.CallOption) (*Project, error) {
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
	ret := &Project{
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
	//   "description": "Fetch the representation of an existing Project.",
	//   "httpMethod": "GET",
	//   "id": "dns.projects.get",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}",
	//   "response": {
	//     "$ref": "Project"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.resourceRecordSets.list":

type ResourceRecordSetsListCall struct {
	s            *Service
	project      string
	managedZone  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Enumerate ResourceRecordSets that have been created but not yet
// deleted.
func (r *ResourceRecordSetsService) List(project string, managedZone string) *ResourceRecordSetsListCall {
	c := &ResourceRecordSetsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.managedZone = managedZone
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to be returned. If unspecified, the server will decide how
// many results to return.
func (c *ResourceRecordSetsListCall) MaxResults(maxResults int64) *ResourceRecordSetsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// Name sets the optional parameter "name": Restricts the list to return
// only records with this fully qualified domain name.
func (c *ResourceRecordSetsListCall) Name(name string) *ResourceRecordSetsListCall {
	c.urlParams_.Set("name", name)
	return c
}

// PageToken sets the optional parameter "pageToken": A tag returned by
// a previous list request that was truncated. Use this parameter to
// continue a previous list request.
func (c *ResourceRecordSetsListCall) PageToken(pageToken string) *ResourceRecordSetsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Type sets the optional parameter "type": Restricts the list to return
// only records of this type. If present, the "name" parameter must also
// be present.
func (c *ResourceRecordSetsListCall) Type(type_ string) *ResourceRecordSetsListCall {
	c.urlParams_.Set("type", type_)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ResourceRecordSetsListCall) Fields(s ...googleapi.Field) *ResourceRecordSetsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ResourceRecordSetsListCall) IfNoneMatch(entityTag string) *ResourceRecordSetsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ResourceRecordSetsListCall) Context(ctx context.Context) *ResourceRecordSetsListCall {
	c.ctx_ = ctx
	return c
}

func (c *ResourceRecordSetsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/rrsets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":     c.project,
		"managedZone": c.managedZone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dns.resourceRecordSets.list" call.
// Exactly one of *ResourceRecordSetsListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ResourceRecordSetsListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ResourceRecordSetsListCall) Do(opts ...googleapi.CallOption) (*ResourceRecordSetsListResponse, error) {
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
	ret := &ResourceRecordSetsListResponse{
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
	//   "description": "Enumerate ResourceRecordSets that have been created but not yet deleted.",
	//   "httpMethod": "GET",
	//   "id": "dns.resourceRecordSets.list",
	//   "parameterOrder": [
	//     "project",
	//     "managedZone"
	//   ],
	//   "parameters": {
	//     "managedZone": {
	//       "description": "Identifies the managed zone addressed by this request. Can be the managed zone name or id.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Optional. Maximum number of results to be returned. If unspecified, the server will decide how many results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "name": {
	//       "description": "Restricts the list to return only records with this fully qualified domain name.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Optional. A tag returned by a previous list request that was truncated. Use this parameter to continue a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Identifies the project addressed by this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "type": {
	//       "description": "Restricts the list to return only records of this type. If present, the \"name\" parameter must also be present.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/managedZones/{managedZone}/rrsets",
	//   "response": {
	//     "$ref": "ResourceRecordSetsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ResourceRecordSetsListCall) Pages(ctx context.Context, f func(*ResourceRecordSetsListResponse) error) error {
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
