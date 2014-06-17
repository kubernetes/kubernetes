// Package coordinate provides access to the Google Maps Coordinate API.
//
// See https://developers.google.com/coordinate/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/coordinate/v1"
//   ...
//   coordinateService, err := coordinate.New(oauthHttpClient)
package coordinate

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

const apiId = "coordinate:v1"
const apiName = "coordinate"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/coordinate/v1/teams/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Google Maps Coordinate jobs
	CoordinateScope = "https://www.googleapis.com/auth/coordinate"

	// View your Google Coordinate jobs
	CoordinateReadonlyScope = "https://www.googleapis.com/auth/coordinate.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.CustomFieldDef = NewCustomFieldDefService(s)
	s.Jobs = NewJobsService(s)
	s.Location = NewLocationService(s)
	s.Schedule = NewScheduleService(s)
	s.Worker = NewWorkerService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	CustomFieldDef *CustomFieldDefService

	Jobs *JobsService

	Location *LocationService

	Schedule *ScheduleService

	Worker *WorkerService
}

func NewCustomFieldDefService(s *Service) *CustomFieldDefService {
	rs := &CustomFieldDefService{s: s}
	return rs
}

type CustomFieldDefService struct {
	s *Service
}

func NewJobsService(s *Service) *JobsService {
	rs := &JobsService{s: s}
	return rs
}

type JobsService struct {
	s *Service
}

func NewLocationService(s *Service) *LocationService {
	rs := &LocationService{s: s}
	return rs
}

type LocationService struct {
	s *Service
}

func NewScheduleService(s *Service) *ScheduleService {
	rs := &ScheduleService{s: s}
	return rs
}

type ScheduleService struct {
	s *Service
}

func NewWorkerService(s *Service) *WorkerService {
	rs := &WorkerService{s: s}
	return rs
}

type WorkerService struct {
	s *Service
}

type CustomField struct {
	// CustomFieldId: Custom field id.
	CustomFieldId int64 `json:"customFieldId,omitempty,string"`

	// Kind: Identifies this object as a custom field.
	Kind string `json:"kind,omitempty"`

	// Value: Custom field value.
	Value string `json:"value,omitempty"`
}

type CustomFieldDef struct {
	// Enabled: Whether the field is enabled.
	Enabled bool `json:"enabled,omitempty"`

	// Id: Custom field id.
	Id int64 `json:"id,omitempty,string"`

	// Kind: Identifies this object as a custom field definition.
	Kind string `json:"kind,omitempty"`

	// Name: Custom field name.
	Name string `json:"name,omitempty"`

	// RequiredForCheckout: Whether the field is required for checkout.
	RequiredForCheckout bool `json:"requiredForCheckout,omitempty"`

	// Type: Custom field type.
	Type string `json:"type,omitempty"`
}

type CustomFieldDefListResponse struct {
	// Items: Collection of custom field definitions in a team.
	Items []*CustomFieldDef `json:"items,omitempty"`

	// Kind: Identifies this object as a collection of custom field
	// definitions in a team.
	Kind string `json:"kind,omitempty"`
}

type CustomFields struct {
	// CustomField: Collection of custom fields.
	CustomField []*CustomField `json:"customField,omitempty"`

	// Kind: Identifies this object as a collection of custom fields.
	Kind string `json:"kind,omitempty"`
}

type Job struct {
	// Id: Job id.
	Id uint64 `json:"id,omitempty,string"`

	// JobChange: List of job changes since it was created. The first change
	// corresponds to the state of the job when it was created.
	JobChange []*JobChange `json:"jobChange,omitempty"`

	// Kind: Identifies this object as a job.
	Kind string `json:"kind,omitempty"`

	// State: Current job state.
	State *JobState `json:"state,omitempty"`
}

type JobChange struct {
	// Kind: Identifies this object as a job change.
	Kind string `json:"kind,omitempty"`

	// State: Change applied to the job. Only the fields that were changed
	// are set.
	State *JobState `json:"state,omitempty"`

	// Timestamp: Time at which this change was applied.
	Timestamp uint64 `json:"timestamp,omitempty,string"`
}

type JobListResponse struct {
	// Items: Jobs in the collection.
	Items []*Job `json:"items,omitempty"`

	// Kind: Identifies this object as a list of jobs.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to provide to get the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type JobState struct {
	// Assignee: Email address of the assignee.
	Assignee string `json:"assignee,omitempty"`

	// CustomFields: Custom fields.
	CustomFields *CustomFields `json:"customFields,omitempty"`

	// CustomerName: Customer name.
	CustomerName string `json:"customerName,omitempty"`

	// CustomerPhoneNumber: Customer phone number.
	CustomerPhoneNumber string `json:"customerPhoneNumber,omitempty"`

	// Kind: Identifies this object as a job state.
	Kind string `json:"kind,omitempty"`

	// Location: Job location.
	Location *Location `json:"location,omitempty"`

	// Note: Note added to the job.
	Note []string `json:"note,omitempty"`

	// Progress: Job progress.
	Progress string `json:"progress,omitempty"`

	// Title: Job title.
	Title string `json:"title,omitempty"`
}

type Location struct {
	// AddressLine: Address.
	AddressLine []string `json:"addressLine,omitempty"`

	// Kind: Identifies this object as a location.
	Kind string `json:"kind,omitempty"`

	// Lat: Latitude.
	Lat float64 `json:"lat,omitempty"`

	// Lng: Longitude.
	Lng float64 `json:"lng,omitempty"`
}

type LocationListResponse struct {
	// Items: Locations in the collection.
	Items []*LocationRecord `json:"items,omitempty"`

	// Kind: Identifies this object as a list of locations.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to provide to get the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TokenPagination: Pagination information for token pagination.
	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`
}

type LocationRecord struct {
	// CollectionTime: The collection time in milliseconds since the epoch.
	CollectionTime int64 `json:"collectionTime,omitempty,string"`

	// ConfidenceRadius: The location accuracy in meters. This is the radius
	// of a 95% confidence interval around the location measurement.
	ConfidenceRadius float64 `json:"confidenceRadius,omitempty"`

	// Kind: Identifies this object as a location.
	Kind string `json:"kind,omitempty"`

	// Latitude: Latitude.
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: Longitude.
	Longitude float64 `json:"longitude,omitempty"`
}

type Schedule struct {
	// AllDay: Whether the job is scheduled for the whole day. Time of day
	// in start/end times is ignored if this is true.
	AllDay bool `json:"allDay,omitempty"`

	// Duration: Job duration in milliseconds.
	Duration uint64 `json:"duration,omitempty,string"`

	// EndTime: Scheduled end time in milliseconds since epoch.
	EndTime uint64 `json:"endTime,omitempty,string"`

	// Kind: Identifies this object as a job schedule.
	Kind string `json:"kind,omitempty"`

	// StartTime: Scheduled start time in milliseconds since epoch.
	StartTime uint64 `json:"startTime,omitempty,string"`
}

type TokenPagination struct {
	// Kind: Identifies this object as pagination information.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to provide to get the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PreviousPageToken: A token to provide to get the previous page of
	// results.
	PreviousPageToken string `json:"previousPageToken,omitempty"`
}

type Worker struct {
	// Id: Worker email address.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this object as a worker.
	Kind string `json:"kind,omitempty"`
}

type WorkerListResponse struct {
	// Items: Workers in the collection.
	Items []*Worker `json:"items,omitempty"`

	// Kind: Identifies this object as a list of workers.
	Kind string `json:"kind,omitempty"`
}

// method id "coordinate.customFieldDef.list":

type CustomFieldDefListCall struct {
	s      *Service
	teamId string
	opt_   map[string]interface{}
}

// List: Retrieves a list of custom field definitions for a team.
func (r *CustomFieldDefService) List(teamId string) *CustomFieldDefListCall {
	c := &CustomFieldDefListCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	return c
}

func (c *CustomFieldDefListCall) Do() (*CustomFieldDefListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/custom_fields")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
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
	ret := new(CustomFieldDefListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of custom field definitions for a team.",
	//   "httpMethod": "GET",
	//   "id": "coordinate.customFieldDef.list",
	//   "parameterOrder": [
	//     "teamId"
	//   ],
	//   "parameters": {
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/custom_fields",
	//   "response": {
	//     "$ref": "CustomFieldDefListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}

// method id "coordinate.jobs.get":

type JobsGetCall struct {
	s      *Service
	teamId string
	jobId  uint64
	opt_   map[string]interface{}
}

// Get: Retrieves a job, including all the changes made to the job.
func (r *JobsService) Get(teamId string, jobId uint64) *JobsGetCall {
	c := &JobsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.jobId = jobId
	return c
}

func (c *JobsGetCall) Do() (*Job, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs/{jobId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{jobId}", strconv.FormatUint(c.jobId, 10), 1)
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
	ret := new(Job)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a job, including all the changes made to the job.",
	//   "httpMethod": "GET",
	//   "id": "coordinate.jobs.get",
	//   "parameterOrder": [
	//     "teamId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "Job number",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}

// method id "coordinate.jobs.insert":

type JobsInsertCall struct {
	s       *Service
	teamId  string
	address string
	lat     float64
	lng     float64
	title   string
	job     *Job
	opt_    map[string]interface{}
}

// Insert: Inserts a new job. Only the state field of the job should be
// set.
func (r *JobsService) Insert(teamId string, address string, lat float64, lng float64, title string, job *Job) *JobsInsertCall {
	c := &JobsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.address = address
	c.lat = lat
	c.lng = lng
	c.title = title
	c.job = job
	return c
}

// Assignee sets the optional parameter "assignee": Assignee email
// address, or empty string to unassign.
func (c *JobsInsertCall) Assignee(assignee string) *JobsInsertCall {
	c.opt_["assignee"] = assignee
	return c
}

// CustomField sets the optional parameter "customField": Map from
// custom field id (from /team//custom_fields) to the field value. For
// example '123=Alice'
func (c *JobsInsertCall) CustomField(customField string) *JobsInsertCall {
	c.opt_["customField"] = customField
	return c
}

// CustomerName sets the optional parameter "customerName": Customer
// name
func (c *JobsInsertCall) CustomerName(customerName string) *JobsInsertCall {
	c.opt_["customerName"] = customerName
	return c
}

// CustomerPhoneNumber sets the optional parameter
// "customerPhoneNumber": Customer phone number
func (c *JobsInsertCall) CustomerPhoneNumber(customerPhoneNumber string) *JobsInsertCall {
	c.opt_["customerPhoneNumber"] = customerPhoneNumber
	return c
}

// Note sets the optional parameter "note": Job note as newline (Unix)
// separated string
func (c *JobsInsertCall) Note(note string) *JobsInsertCall {
	c.opt_["note"] = note
	return c
}

func (c *JobsInsertCall) Do() (*Job, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("address", fmt.Sprintf("%v", c.address))
	params.Set("lat", fmt.Sprintf("%v", c.lat))
	params.Set("lng", fmt.Sprintf("%v", c.lng))
	params.Set("title", fmt.Sprintf("%v", c.title))
	if v, ok := c.opt_["assignee"]; ok {
		params.Set("assignee", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customField"]; ok {
		params.Set("customField", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerName"]; ok {
		params.Set("customerName", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerPhoneNumber"]; ok {
		params.Set("customerPhoneNumber", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["note"]; ok {
		params.Set("note", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
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
	ret := new(Job)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new job. Only the state field of the job should be set.",
	//   "httpMethod": "POST",
	//   "id": "coordinate.jobs.insert",
	//   "parameterOrder": [
	//     "teamId",
	//     "address",
	//     "lat",
	//     "lng",
	//     "title"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "Job address as newline (Unix) separated string",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "assignee": {
	//       "description": "Assignee email address, or empty string to unassign.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customField": {
	//       "description": "Map from custom field id (from /team//custom_fields) to the field value. For example '123=Alice'",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "customerName": {
	//       "description": "Customer name",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customerPhoneNumber": {
	//       "description": "Customer phone number",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "lat": {
	//       "description": "The latitude coordinate of this job's location.",
	//       "format": "double",
	//       "location": "query",
	//       "required": true,
	//       "type": "number"
	//     },
	//     "lng": {
	//       "description": "The longitude coordinate of this job's location.",
	//       "format": "double",
	//       "location": "query",
	//       "required": true,
	//       "type": "number"
	//     },
	//     "note": {
	//       "description": "Job note as newline (Unix) separated string",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "title": {
	//       "description": "Job title",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs",
	//   "request": {
	//     "$ref": "Job"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate"
	//   ]
	// }

}

// method id "coordinate.jobs.list":

type JobsListCall struct {
	s      *Service
	teamId string
	opt_   map[string]interface{}
}

// List: Retrieves jobs created or modified since the given timestamp.
func (r *JobsService) List(teamId string) *JobsListCall {
	c := &JobsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return in one page.
func (c *JobsListCall) MaxResults(maxResults int64) *JobsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// MinModifiedTimestampMs sets the optional parameter
// "minModifiedTimestampMs": Minimum time a job was modified in
// milliseconds since epoch.
func (c *JobsListCall) MinModifiedTimestampMs(minModifiedTimestampMs uint64) *JobsListCall {
	c.opt_["minModifiedTimestampMs"] = minModifiedTimestampMs
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
func (c *JobsListCall) PageToken(pageToken string) *JobsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *JobsListCall) Do() (*JobListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["minModifiedTimestampMs"]; ok {
		params.Set("minModifiedTimestampMs", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
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
	ret := new(JobListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves jobs created or modified since the given timestamp.",
	//   "httpMethod": "GET",
	//   "id": "coordinate.jobs.list",
	//   "parameterOrder": [
	//     "teamId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return in one page.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "minModifiedTimestampMs": {
	//       "description": "Minimum time a job was modified in milliseconds since epoch.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs",
	//   "response": {
	//     "$ref": "JobListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}

// method id "coordinate.jobs.patch":

type JobsPatchCall struct {
	s      *Service
	teamId string
	jobId  uint64
	job    *Job
	opt_   map[string]interface{}
}

// Patch: Updates a job. Fields that are set in the job state will be
// updated. This method supports patch semantics.
func (r *JobsService) Patch(teamId string, jobId uint64, job *Job) *JobsPatchCall {
	c := &JobsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.jobId = jobId
	c.job = job
	return c
}

// Address sets the optional parameter "address": Job address as newline
// (Unix) separated string
func (c *JobsPatchCall) Address(address string) *JobsPatchCall {
	c.opt_["address"] = address
	return c
}

// Assignee sets the optional parameter "assignee": Assignee email
// address, or empty string to unassign.
func (c *JobsPatchCall) Assignee(assignee string) *JobsPatchCall {
	c.opt_["assignee"] = assignee
	return c
}

// CustomField sets the optional parameter "customField": Map from
// custom field id (from /team//custom_fields) to the field value. For
// example '123=Alice'
func (c *JobsPatchCall) CustomField(customField string) *JobsPatchCall {
	c.opt_["customField"] = customField
	return c
}

// CustomerName sets the optional parameter "customerName": Customer
// name
func (c *JobsPatchCall) CustomerName(customerName string) *JobsPatchCall {
	c.opt_["customerName"] = customerName
	return c
}

// CustomerPhoneNumber sets the optional parameter
// "customerPhoneNumber": Customer phone number
func (c *JobsPatchCall) CustomerPhoneNumber(customerPhoneNumber string) *JobsPatchCall {
	c.opt_["customerPhoneNumber"] = customerPhoneNumber
	return c
}

// Lat sets the optional parameter "lat": The latitude coordinate of
// this job's location.
func (c *JobsPatchCall) Lat(lat float64) *JobsPatchCall {
	c.opt_["lat"] = lat
	return c
}

// Lng sets the optional parameter "lng": The longitude coordinate of
// this job's location.
func (c *JobsPatchCall) Lng(lng float64) *JobsPatchCall {
	c.opt_["lng"] = lng
	return c
}

// Note sets the optional parameter "note": Job note as newline (Unix)
// separated string
func (c *JobsPatchCall) Note(note string) *JobsPatchCall {
	c.opt_["note"] = note
	return c
}

// Progress sets the optional parameter "progress": Job progress
func (c *JobsPatchCall) Progress(progress string) *JobsPatchCall {
	c.opt_["progress"] = progress
	return c
}

// Title sets the optional parameter "title": Job title
func (c *JobsPatchCall) Title(title string) *JobsPatchCall {
	c.opt_["title"] = title
	return c
}

func (c *JobsPatchCall) Do() (*Job, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["address"]; ok {
		params.Set("address", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["assignee"]; ok {
		params.Set("assignee", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customField"]; ok {
		params.Set("customField", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerName"]; ok {
		params.Set("customerName", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerPhoneNumber"]; ok {
		params.Set("customerPhoneNumber", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["lat"]; ok {
		params.Set("lat", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["lng"]; ok {
		params.Set("lng", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["note"]; ok {
		params.Set("note", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["progress"]; ok {
		params.Set("progress", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["title"]; ok {
		params.Set("title", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs/{jobId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{jobId}", strconv.FormatUint(c.jobId, 10), 1)
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
	ret := new(Job)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a job. Fields that are set in the job state will be updated. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "coordinate.jobs.patch",
	//   "parameterOrder": [
	//     "teamId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "Job address as newline (Unix) separated string",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "assignee": {
	//       "description": "Assignee email address, or empty string to unassign.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customField": {
	//       "description": "Map from custom field id (from /team//custom_fields) to the field value. For example '123=Alice'",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "customerName": {
	//       "description": "Customer name",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customerPhoneNumber": {
	//       "description": "Customer phone number",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "jobId": {
	//       "description": "Job number",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "lat": {
	//       "description": "The latitude coordinate of this job's location.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "lng": {
	//       "description": "The longitude coordinate of this job's location.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "note": {
	//       "description": "Job note as newline (Unix) separated string",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "progress": {
	//       "description": "Job progress",
	//       "enum": [
	//         "COMPLETED",
	//         "IN_PROGRESS",
	//         "NOT_ACCEPTED",
	//         "NOT_STARTED",
	//         "OBSOLETE"
	//       ],
	//       "enumDescriptions": [
	//         "Completed",
	//         "In progress",
	//         "Not accepted",
	//         "Not started",
	//         "Obsolete"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "title": {
	//       "description": "Job title",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs/{jobId}",
	//   "request": {
	//     "$ref": "Job"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate"
	//   ]
	// }

}

// method id "coordinate.jobs.update":

type JobsUpdateCall struct {
	s      *Service
	teamId string
	jobId  uint64
	job    *Job
	opt_   map[string]interface{}
}

// Update: Updates a job. Fields that are set in the job state will be
// updated.
func (r *JobsService) Update(teamId string, jobId uint64, job *Job) *JobsUpdateCall {
	c := &JobsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.jobId = jobId
	c.job = job
	return c
}

// Address sets the optional parameter "address": Job address as newline
// (Unix) separated string
func (c *JobsUpdateCall) Address(address string) *JobsUpdateCall {
	c.opt_["address"] = address
	return c
}

// Assignee sets the optional parameter "assignee": Assignee email
// address, or empty string to unassign.
func (c *JobsUpdateCall) Assignee(assignee string) *JobsUpdateCall {
	c.opt_["assignee"] = assignee
	return c
}

// CustomField sets the optional parameter "customField": Map from
// custom field id (from /team//custom_fields) to the field value. For
// example '123=Alice'
func (c *JobsUpdateCall) CustomField(customField string) *JobsUpdateCall {
	c.opt_["customField"] = customField
	return c
}

// CustomerName sets the optional parameter "customerName": Customer
// name
func (c *JobsUpdateCall) CustomerName(customerName string) *JobsUpdateCall {
	c.opt_["customerName"] = customerName
	return c
}

// CustomerPhoneNumber sets the optional parameter
// "customerPhoneNumber": Customer phone number
func (c *JobsUpdateCall) CustomerPhoneNumber(customerPhoneNumber string) *JobsUpdateCall {
	c.opt_["customerPhoneNumber"] = customerPhoneNumber
	return c
}

// Lat sets the optional parameter "lat": The latitude coordinate of
// this job's location.
func (c *JobsUpdateCall) Lat(lat float64) *JobsUpdateCall {
	c.opt_["lat"] = lat
	return c
}

// Lng sets the optional parameter "lng": The longitude coordinate of
// this job's location.
func (c *JobsUpdateCall) Lng(lng float64) *JobsUpdateCall {
	c.opt_["lng"] = lng
	return c
}

// Note sets the optional parameter "note": Job note as newline (Unix)
// separated string
func (c *JobsUpdateCall) Note(note string) *JobsUpdateCall {
	c.opt_["note"] = note
	return c
}

// Progress sets the optional parameter "progress": Job progress
func (c *JobsUpdateCall) Progress(progress string) *JobsUpdateCall {
	c.opt_["progress"] = progress
	return c
}

// Title sets the optional parameter "title": Job title
func (c *JobsUpdateCall) Title(title string) *JobsUpdateCall {
	c.opt_["title"] = title
	return c
}

func (c *JobsUpdateCall) Do() (*Job, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["address"]; ok {
		params.Set("address", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["assignee"]; ok {
		params.Set("assignee", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customField"]; ok {
		params.Set("customField", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerName"]; ok {
		params.Set("customerName", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["customerPhoneNumber"]; ok {
		params.Set("customerPhoneNumber", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["lat"]; ok {
		params.Set("lat", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["lng"]; ok {
		params.Set("lng", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["note"]; ok {
		params.Set("note", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["progress"]; ok {
		params.Set("progress", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["title"]; ok {
		params.Set("title", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs/{jobId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{jobId}", strconv.FormatUint(c.jobId, 10), 1)
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
	ret := new(Job)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a job. Fields that are set in the job state will be updated.",
	//   "httpMethod": "PUT",
	//   "id": "coordinate.jobs.update",
	//   "parameterOrder": [
	//     "teamId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "Job address as newline (Unix) separated string",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "assignee": {
	//       "description": "Assignee email address, or empty string to unassign.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customField": {
	//       "description": "Map from custom field id (from /team//custom_fields) to the field value. For example '123=Alice'",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "customerName": {
	//       "description": "Customer name",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "customerPhoneNumber": {
	//       "description": "Customer phone number",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "jobId": {
	//       "description": "Job number",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "lat": {
	//       "description": "The latitude coordinate of this job's location.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "lng": {
	//       "description": "The longitude coordinate of this job's location.",
	//       "format": "double",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "note": {
	//       "description": "Job note as newline (Unix) separated string",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "progress": {
	//       "description": "Job progress",
	//       "enum": [
	//         "COMPLETED",
	//         "IN_PROGRESS",
	//         "NOT_ACCEPTED",
	//         "NOT_STARTED",
	//         "OBSOLETE"
	//       ],
	//       "enumDescriptions": [
	//         "Completed",
	//         "In progress",
	//         "Not accepted",
	//         "Not started",
	//         "Obsolete"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "title": {
	//       "description": "Job title",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs/{jobId}",
	//   "request": {
	//     "$ref": "Job"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate"
	//   ]
	// }

}

// method id "coordinate.location.list":

type LocationListCall struct {
	s                *Service
	teamId           string
	workerEmail      string
	startTimestampMs uint64
	opt_             map[string]interface{}
}

// List: Retrieves a list of locations for a worker.
func (r *LocationService) List(teamId string, workerEmail string, startTimestampMs uint64) *LocationListCall {
	c := &LocationListCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.workerEmail = workerEmail
	c.startTimestampMs = startTimestampMs
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return in one page.
func (c *LocationListCall) MaxResults(maxResults int64) *LocationListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
func (c *LocationListCall) PageToken(pageToken string) *LocationListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *LocationListCall) Do() (*LocationListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("startTimestampMs", fmt.Sprintf("%v", c.startTimestampMs))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/workers/{workerEmail}/locations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{workerEmail}", url.QueryEscape(c.workerEmail), 1)
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
	ret := new(LocationListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of locations for a worker.",
	//   "httpMethod": "GET",
	//   "id": "coordinate.location.list",
	//   "parameterOrder": [
	//     "teamId",
	//     "workerEmail",
	//     "startTimestampMs"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return in one page.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startTimestampMs": {
	//       "description": "Start timestamp in milliseconds since the epoch.",
	//       "format": "uint64",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "workerEmail": {
	//       "description": "Worker email address.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/workers/{workerEmail}/locations",
	//   "response": {
	//     "$ref": "LocationListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}

// method id "coordinate.schedule.get":

type ScheduleGetCall struct {
	s      *Service
	teamId string
	jobId  uint64
	opt_   map[string]interface{}
}

// Get: Retrieves the schedule for a job.
func (r *ScheduleService) Get(teamId string, jobId uint64) *ScheduleGetCall {
	c := &ScheduleGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.jobId = jobId
	return c
}

func (c *ScheduleGetCall) Do() (*Schedule, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs/{jobId}/schedule")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{jobId}", strconv.FormatUint(c.jobId, 10), 1)
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
	ret := new(Schedule)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the schedule for a job.",
	//   "httpMethod": "GET",
	//   "id": "coordinate.schedule.get",
	//   "parameterOrder": [
	//     "teamId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "Job number",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs/{jobId}/schedule",
	//   "response": {
	//     "$ref": "Schedule"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}

// method id "coordinate.schedule.patch":

type SchedulePatchCall struct {
	s        *Service
	teamId   string
	jobId    uint64
	schedule *Schedule
	opt_     map[string]interface{}
}

// Patch: Replaces the schedule of a job with the provided schedule.
// This method supports patch semantics.
func (r *ScheduleService) Patch(teamId string, jobId uint64, schedule *Schedule) *SchedulePatchCall {
	c := &SchedulePatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.jobId = jobId
	c.schedule = schedule
	return c
}

// AllDay sets the optional parameter "allDay": Whether the job is
// scheduled for the whole day. Time of day in start/end times is
// ignored if this is true.
func (c *SchedulePatchCall) AllDay(allDay bool) *SchedulePatchCall {
	c.opt_["allDay"] = allDay
	return c
}

// Duration sets the optional parameter "duration": Job duration in
// milliseconds.
func (c *SchedulePatchCall) Duration(duration uint64) *SchedulePatchCall {
	c.opt_["duration"] = duration
	return c
}

// EndTime sets the optional parameter "endTime": Scheduled end time in
// milliseconds since epoch.
func (c *SchedulePatchCall) EndTime(endTime uint64) *SchedulePatchCall {
	c.opt_["endTime"] = endTime
	return c
}

// StartTime sets the optional parameter "startTime": Scheduled start
// time in milliseconds since epoch.
func (c *SchedulePatchCall) StartTime(startTime uint64) *SchedulePatchCall {
	c.opt_["startTime"] = startTime
	return c
}

func (c *SchedulePatchCall) Do() (*Schedule, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.schedule)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["allDay"]; ok {
		params.Set("allDay", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["duration"]; ok {
		params.Set("duration", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["endTime"]; ok {
		params.Set("endTime", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startTime"]; ok {
		params.Set("startTime", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs/{jobId}/schedule")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{jobId}", strconv.FormatUint(c.jobId, 10), 1)
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
	ret := new(Schedule)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Replaces the schedule of a job with the provided schedule. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "coordinate.schedule.patch",
	//   "parameterOrder": [
	//     "teamId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "allDay": {
	//       "description": "Whether the job is scheduled for the whole day. Time of day in start/end times is ignored if this is true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "duration": {
	//       "description": "Job duration in milliseconds.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "endTime": {
	//       "description": "Scheduled end time in milliseconds since epoch.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "jobId": {
	//       "description": "Job number",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startTime": {
	//       "description": "Scheduled start time in milliseconds since epoch.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs/{jobId}/schedule",
	//   "request": {
	//     "$ref": "Schedule"
	//   },
	//   "response": {
	//     "$ref": "Schedule"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate"
	//   ]
	// }

}

// method id "coordinate.schedule.update":

type ScheduleUpdateCall struct {
	s        *Service
	teamId   string
	jobId    uint64
	schedule *Schedule
	opt_     map[string]interface{}
}

// Update: Replaces the schedule of a job with the provided schedule.
func (r *ScheduleService) Update(teamId string, jobId uint64, schedule *Schedule) *ScheduleUpdateCall {
	c := &ScheduleUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	c.jobId = jobId
	c.schedule = schedule
	return c
}

// AllDay sets the optional parameter "allDay": Whether the job is
// scheduled for the whole day. Time of day in start/end times is
// ignored if this is true.
func (c *ScheduleUpdateCall) AllDay(allDay bool) *ScheduleUpdateCall {
	c.opt_["allDay"] = allDay
	return c
}

// Duration sets the optional parameter "duration": Job duration in
// milliseconds.
func (c *ScheduleUpdateCall) Duration(duration uint64) *ScheduleUpdateCall {
	c.opt_["duration"] = duration
	return c
}

// EndTime sets the optional parameter "endTime": Scheduled end time in
// milliseconds since epoch.
func (c *ScheduleUpdateCall) EndTime(endTime uint64) *ScheduleUpdateCall {
	c.opt_["endTime"] = endTime
	return c
}

// StartTime sets the optional parameter "startTime": Scheduled start
// time in milliseconds since epoch.
func (c *ScheduleUpdateCall) StartTime(startTime uint64) *ScheduleUpdateCall {
	c.opt_["startTime"] = startTime
	return c
}

func (c *ScheduleUpdateCall) Do() (*Schedule, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.schedule)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["allDay"]; ok {
		params.Set("allDay", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["duration"]; ok {
		params.Set("duration", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["endTime"]; ok {
		params.Set("endTime", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startTime"]; ok {
		params.Set("startTime", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/jobs/{jobId}/schedule")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{jobId}", strconv.FormatUint(c.jobId, 10), 1)
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
	ret := new(Schedule)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Replaces the schedule of a job with the provided schedule.",
	//   "httpMethod": "PUT",
	//   "id": "coordinate.schedule.update",
	//   "parameterOrder": [
	//     "teamId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "allDay": {
	//       "description": "Whether the job is scheduled for the whole day. Time of day in start/end times is ignored if this is true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "duration": {
	//       "description": "Job duration in milliseconds.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "endTime": {
	//       "description": "Scheduled end time in milliseconds since epoch.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "jobId": {
	//       "description": "Job number",
	//       "format": "uint64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startTime": {
	//       "description": "Scheduled start time in milliseconds since epoch.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/jobs/{jobId}/schedule",
	//   "request": {
	//     "$ref": "Schedule"
	//   },
	//   "response": {
	//     "$ref": "Schedule"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate"
	//   ]
	// }

}

// method id "coordinate.worker.list":

type WorkerListCall struct {
	s      *Service
	teamId string
	opt_   map[string]interface{}
}

// List: Retrieves a list of workers in a team.
func (r *WorkerService) List(teamId string) *WorkerListCall {
	c := &WorkerListCall{s: r.s, opt_: make(map[string]interface{})}
	c.teamId = teamId
	return c
}

func (c *WorkerListCall) Do() (*WorkerListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{teamId}/workers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{teamId}", url.QueryEscape(c.teamId), 1)
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
	ret := new(WorkerListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of workers in a team.",
	//   "httpMethod": "GET",
	//   "id": "coordinate.worker.list",
	//   "parameterOrder": [
	//     "teamId"
	//   ],
	//   "parameters": {
	//     "teamId": {
	//       "description": "Team ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{teamId}/workers",
	//   "response": {
	//     "$ref": "WorkerListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}
