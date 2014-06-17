// Package dns provides access to the Google Cloud DNS API.
//
// See https://developers.google.com/cloud-dns
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/dns/v1beta1"
//   ...
//   dnsService, err := dns.New(oauthHttpClient)
package dns

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

const apiId = "dns:v1beta1"
const apiName = "dns"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/dns/v1beta1/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

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
	client   *http.Client
	BasePath string // API endpoint base URL

	Changes *ChangesService

	ManagedZones *ManagedZonesService

	Projects *ProjectsService

	ResourceRecordSets *ResourceRecordSetsService
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

	// StartTime: The time that this operation was started by the server.
	// This is in RFC3339 text format.
	StartTime string `json:"startTime,omitempty"`

	// Status: Status of the operation. Can be one of the following:
	// "PENDING" or "DONE" (output only).
	Status string `json:"status,omitempty"`
}

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
	// In this way you can retrieve the complete contents
	// of even very large collections one page at a time. However, if the
	// contents of the collection change between the first and last
	// paginated list request, the set of all elements returned will be an
	// inconsistent view of the collection. There is no way to retrieve a
	// "snapshot" of collections larger than the maximum page size.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type ManagedZone struct {
	// CreationTime: The time that this resource was created on the server.
	// This is in RFC3339 text format. Output only.
	CreationTime string `json:"creationTime,omitempty"`

	// Description: A string to associate with this resource for the user's
	// convenience. Has no effect on the managed zone's function.
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
	// project.
	Name string `json:"name,omitempty"`

	// NameServers: Delegate your managed_zone to these virtual name
	// servers; defined by the server (output only)
	NameServers []string `json:"nameServers,omitempty"`
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
	// In this way you can retrieve the complete contents of
	// even very large collections one page at a time. However, if the
	// contents of the collection change between the first and last
	// paginated list request, the set of all elements returned will be an
	// inconsistent view of the collection. There is no way to retrieve a
	// consistent snapshot of a collection larger than the maximum page
	// size.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

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
}

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
}

type ResourceRecordSet struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "dns#resourceRecordSet".
	Kind string `json:"kind,omitempty"`

	// Name: For example, www.example.com.
	Name string `json:"name,omitempty"`

	// Rrdatas: As defined in RFC 1035 (section 5) and RFC 1034 (section
	// 3.6.1)
	Rrdatas []string `json:"rrdatas,omitempty"`

	// Ttl: Number of seconds that this ResourceRecordSet can be cached by
	// resolvers.
	Ttl int64 `json:"ttl,omitempty"`

	// Type: One of A, AAAA, SOA, MX, NS, TXT
	Type string `json:"type,omitempty"`
}

type ResourceRecordSetsListResponse struct {
	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The presence of this field indicates that there exist
	// more results following your last page of results in pagination order.
	// To fetch them, make another list request using this value as your
	// pagination token.
	//
	// In this way you can retrieve the complete contents
	// of even very large collections one page at a time. However, if the
	// contents of the collection change between the first and last
	// paginated list request, the set of all elements returned will be an
	// inconsistent view of the collection. There is no way to retrieve a
	// consistent snapshot of a collection larger than the maximum page
	// size.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Rrsets: The resource record set resources.
	Rrsets []*ResourceRecordSet `json:"rrsets,omitempty"`
}

// method id "dns.changes.create":

type ChangesCreateCall struct {
	s           *Service
	project     string
	managedZone string
	change      *Change
	opt_        map[string]interface{}
}

// Create: Atomically update the ResourceRecordSet collection.
func (r *ChangesService) Create(project string, managedZone string, change *Change) *ChangesCreateCall {
	c := &ChangesCreateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.managedZone = managedZone
	c.change = change
	return c
}

func (c *ChangesCreateCall) Do() (*Change, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.change)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/changes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{managedZone}", url.QueryEscape(c.managedZone), 1)
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
	ret := new(Change)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	s           *Service
	project     string
	managedZone string
	changeId    string
	opt_        map[string]interface{}
}

// Get: Fetch the representation of an existing Change.
func (r *ChangesService) Get(project string, managedZone string, changeId string) *ChangesGetCall {
	c := &ChangesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.managedZone = managedZone
	c.changeId = changeId
	return c
}

func (c *ChangesGetCall) Do() (*Change, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/changes/{changeId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{managedZone}", url.QueryEscape(c.managedZone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{changeId}", url.QueryEscape(c.changeId), 1)
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
	ret := new(Change)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.changes.list":

type ChangesListCall struct {
	s           *Service
	project     string
	managedZone string
	opt_        map[string]interface{}
}

// List: Enumerate Changes to a ResourceRecordSet collection.
func (r *ChangesService) List(project string, managedZone string) *ChangesListCall {
	c := &ChangesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.managedZone = managedZone
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to be returned. If unspecified, the server will decide how
// many results to return.
func (c *ChangesListCall) MaxResults(maxResults int64) *ChangesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A tag returned by
// a previous list request that was truncated. Use this parameter to
// continue a previous list request.
func (c *ChangesListCall) PageToken(pageToken string) *ChangesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// SortBy sets the optional parameter "sortBy": Sorting criterion. The
// only supported value is change sequence.
func (c *ChangesListCall) SortBy(sortBy string) *ChangesListCall {
	c.opt_["sortBy"] = sortBy
	return c
}

// SortOrder sets the optional parameter "sortOrder": Sorting order
// direction: 'ascending' or 'descending'.
func (c *ChangesListCall) SortOrder(sortOrder string) *ChangesListCall {
	c.opt_["sortOrder"] = sortOrder
	return c
}

func (c *ChangesListCall) Do() (*ChangesListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortBy"]; ok {
		params.Set("sortBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortOrder"]; ok {
		params.Set("sortOrder", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/changes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{managedZone}", url.QueryEscape(c.managedZone), 1)
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
	ret := new(ChangesListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.managedZones.create":

type ManagedZonesCreateCall struct {
	s           *Service
	project     string
	managedzone *ManagedZone
	opt_        map[string]interface{}
}

// Create: Create a new ManagedZone.
func (r *ManagedZonesService) Create(project string, managedzone *ManagedZone) *ManagedZonesCreateCall {
	c := &ManagedZonesCreateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.managedzone = managedzone
	return c
}

func (c *ManagedZonesCreateCall) Do() (*ManagedZone, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.managedzone)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
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
	ret := new(ManagedZone)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_        map[string]interface{}
}

// Delete: Delete a previously created ManagedZone.
func (r *ManagedZonesService) Delete(project string, managedZone string) *ManagedZonesDeleteCall {
	c := &ManagedZonesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.managedZone = managedZone
	return c
}

func (c *ManagedZonesDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{managedZone}", url.QueryEscape(c.managedZone), 1)
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
	s           *Service
	project     string
	managedZone string
	opt_        map[string]interface{}
}

// Get: Fetch the representation of an existing ManagedZone.
func (r *ManagedZonesService) Get(project string, managedZone string) *ManagedZonesGetCall {
	c := &ManagedZonesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.managedZone = managedZone
	return c
}

func (c *ManagedZonesGetCall) Do() (*ManagedZone, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{managedZone}", url.QueryEscape(c.managedZone), 1)
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
	ret := new(ManagedZone)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.managedZones.list":

type ManagedZonesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Enumerate ManagedZones that have been created but not yet
// deleted.
func (r *ManagedZonesService) List(project string) *ManagedZonesListCall {
	c := &ManagedZonesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to be returned. If unspecified, the server will decide how
// many results to return.
func (c *ManagedZonesListCall) MaxResults(maxResults int64) *ManagedZonesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A tag returned by
// a previous list request that was truncated. Use this parameter to
// continue a previous list request.
func (c *ManagedZonesListCall) PageToken(pageToken string) *ManagedZonesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *ManagedZonesListCall) Do() (*ManagedZonesListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
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
	ret := new(ManagedZonesListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.projects.get":

type ProjectsGetCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// Get: Fetch the representation of an existing Project.
func (r *ProjectsService) Get(project string) *ProjectsGetCall {
	c := &ProjectsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

func (c *ProjectsGetCall) Do() (*Project, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
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
	ret := new(Project)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}

// method id "dns.resourceRecordSets.list":

type ResourceRecordSetsListCall struct {
	s           *Service
	project     string
	managedZone string
	opt_        map[string]interface{}
}

// List: Enumerate ResourceRecordSets that have been created but not yet
// deleted.
func (r *ResourceRecordSetsService) List(project string, managedZone string) *ResourceRecordSetsListCall {
	c := &ResourceRecordSetsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.managedZone = managedZone
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to be returned. If unspecified, the server will decide how
// many results to return.
func (c *ResourceRecordSetsListCall) MaxResults(maxResults int64) *ResourceRecordSetsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Name sets the optional parameter "name": Restricts the list to return
// only records with this fully qualified domain name.
func (c *ResourceRecordSetsListCall) Name(name string) *ResourceRecordSetsListCall {
	c.opt_["name"] = name
	return c
}

// PageToken sets the optional parameter "pageToken": A tag returned by
// a previous list request that was truncated. Use this parameter to
// continue a previous list request.
func (c *ResourceRecordSetsListCall) PageToken(pageToken string) *ResourceRecordSetsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Type sets the optional parameter "type": Restricts the list to return
// only records of this type. If present, the "name" parameter must also
// be present.
func (c *ResourceRecordSetsListCall) Type(type_ string) *ResourceRecordSetsListCall {
	c.opt_["type"] = type_
	return c
}

func (c *ResourceRecordSetsListCall) Do() (*ResourceRecordSetsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["name"]; ok {
		params.Set("name", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["type"]; ok {
		params.Set("type", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/managedZones/{managedZone}/rrsets")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{managedZone}", url.QueryEscape(c.managedZone), 1)
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
	ret := new(ResourceRecordSetsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/ndev.clouddns.readonly",
	//     "https://www.googleapis.com/auth/ndev.clouddns.readwrite"
	//   ]
	// }

}
