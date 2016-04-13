// Package coordinate provides access to the Google Maps Coordinate API.
//
// See https://developers.google.com/coordinate/
//
// Usage example:
//
//   import "google.golang.org/api/coordinate/v1"
//   ...
//   coordinateService, err := coordinate.New(oauthHttpClient)
package coordinate // import "google.golang.org/api/coordinate/v1"

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

const apiId = "coordinate:v1"
const apiName = "coordinate"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/coordinate/v1/"

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
	s.Team = NewTeamService(s)
	s.Worker = NewWorkerService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	CustomFieldDef *CustomFieldDefService

	Jobs *JobsService

	Location *LocationService

	Schedule *ScheduleService

	Team *TeamService

	Worker *WorkerService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
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

func NewTeamService(s *Service) *TeamService {
	rs := &TeamService{s: s}
	return rs
}

type TeamService struct {
	s *Service
}

func NewWorkerService(s *Service) *WorkerService {
	rs := &WorkerService{s: s}
	return rs
}

type WorkerService struct {
	s *Service
}

// CustomField: Custom field.
type CustomField struct {
	// CustomFieldId: Custom field id.
	CustomFieldId int64 `json:"customFieldId,omitempty,string"`

	// Kind: Identifies this object as a custom field.
	Kind string `json:"kind,omitempty"`

	// Value: Custom field value.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CustomFieldId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CustomField) MarshalJSON() ([]byte, error) {
	type noMethod CustomField
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CustomFieldDef: Custom field definition.
type CustomFieldDef struct {
	// Enabled: Whether the field is enabled.
	Enabled bool `json:"enabled,omitempty"`

	// Enumitems: List of enum items for this custom field. Populated only
	// if the field type is enum. Enum fields appear as 'lists' in the
	// Coordinate web and mobile UI.
	Enumitems []*EnumItemDef `json:"enumitems,omitempty"`

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

	// ForceSendFields is a list of field names (e.g. "Enabled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CustomFieldDef) MarshalJSON() ([]byte, error) {
	type noMethod CustomFieldDef
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CustomFieldDefListResponse: Collection of custom field definitions
// for a team.
type CustomFieldDefListResponse struct {
	// Items: Collection of custom field definitions in a team.
	Items []*CustomFieldDef `json:"items,omitempty"`

	// Kind: Identifies this object as a collection of custom field
	// definitions in a team.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CustomFieldDefListResponse) MarshalJSON() ([]byte, error) {
	type noMethod CustomFieldDefListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CustomFields: Collection of custom fields.
type CustomFields struct {
	// CustomField: Collection of custom fields.
	CustomField []*CustomField `json:"customField,omitempty"`

	// Kind: Identifies this object as a collection of custom fields.
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CustomField") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CustomFields) MarshalJSON() ([]byte, error) {
	type noMethod CustomFields
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// EnumItemDef: Enum Item definition.
type EnumItemDef struct {
	// Active: Whether the enum item is active. Jobs may contain inactive
	// enum values; however, setting an enum to an inactive value when
	// creating or updating a job will result in a 500 error.
	Active bool `json:"active,omitempty"`

	// Kind: Identifies this object as an enum item definition.
	Kind string `json:"kind,omitempty"`

	// Value: Custom field value.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Active") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EnumItemDef) MarshalJSON() ([]byte, error) {
	type noMethod EnumItemDef
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Job: A job.
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

func (s *Job) MarshalJSON() ([]byte, error) {
	type noMethod Job
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// JobChange: Change to a job. For example assigning the job to a
// different worker.
type JobChange struct {
	// Kind: Identifies this object as a job change.
	Kind string `json:"kind,omitempty"`

	// State: Change applied to the job. Only the fields that were changed
	// are set.
	State *JobState `json:"state,omitempty"`

	// Timestamp: Time at which this change was applied.
	Timestamp uint64 `json:"timestamp,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *JobChange) MarshalJSON() ([]byte, error) {
	type noMethod JobChange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// JobListResponse: Response from a List Jobs request.
type JobListResponse struct {
	// Items: Jobs in the collection.
	Items []*Job `json:"items,omitempty"`

	// Kind: Identifies this object as a list of jobs.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to provide to get the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *JobListResponse) MarshalJSON() ([]byte, error) {
	type noMethod JobListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// JobState: Current state of a job.
type JobState struct {
	// Assignee: Email address of the assignee, or the string "DELETED_USER"
	// if the account is no longer available.
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

	// ForceSendFields is a list of field names (e.g. "Assignee") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *JobState) MarshalJSON() ([]byte, error) {
	type noMethod JobState
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Location: Location of a job.
type Location struct {
	// AddressLine: Address.
	AddressLine []string `json:"addressLine,omitempty"`

	// Kind: Identifies this object as a location.
	Kind string `json:"kind,omitempty"`

	// Lat: Latitude.
	Lat float64 `json:"lat,omitempty"`

	// Lng: Longitude.
	Lng float64 `json:"lng,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AddressLine") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Location) MarshalJSON() ([]byte, error) {
	type noMethod Location
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LocationListResponse: Response from a List Locations request.
type LocationListResponse struct {
	// Items: Locations in the collection.
	Items []*LocationRecord `json:"items,omitempty"`

	// Kind: Identifies this object as a list of locations.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to provide to get the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TokenPagination: Pagination information for token pagination.
	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LocationListResponse) MarshalJSON() ([]byte, error) {
	type noMethod LocationListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LocationRecord: Recorded location of a worker.
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

	// ForceSendFields is a list of field names (e.g. "CollectionTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LocationRecord) MarshalJSON() ([]byte, error) {
	type noMethod LocationRecord
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Schedule: Job schedule.
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AllDay") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Schedule) MarshalJSON() ([]byte, error) {
	type noMethod Schedule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Team: A Coordinate team.
type Team struct {
	// Id: Team id, as found in a coordinate team url e.g.
	// https://coordinate.google.com/f/xyz where "xyz" is the team id.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this object as a team.
	Kind string `json:"kind,omitempty"`

	// Name: Team name
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Team) MarshalJSON() ([]byte, error) {
	type noMethod Team
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TeamListResponse: Response from a List Teams request.
type TeamListResponse struct {
	// Items: Teams in the collection.
	Items []*Team `json:"items,omitempty"`

	// Kind: Identifies this object as a list of teams.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TeamListResponse) MarshalJSON() ([]byte, error) {
	type noMethod TeamListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TokenPagination: Pagination information.
type TokenPagination struct {
	// Kind: Identifies this object as pagination information.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to provide to get the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PreviousPageToken: A token to provide to get the previous page of
	// results.
	PreviousPageToken string `json:"previousPageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TokenPagination) MarshalJSON() ([]byte, error) {
	type noMethod TokenPagination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Worker: A worker in a Coordinate team.
type Worker struct {
	// Id: Worker email address. If a worker has been deleted from your
	// team, the email address will appear as DELETED_USER.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this object as a worker.
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Worker) MarshalJSON() ([]byte, error) {
	type noMethod Worker
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// WorkerListResponse: Response from a List Workers request.
type WorkerListResponse struct {
	// Items: Workers in the collection.
	Items []*Worker `json:"items,omitempty"`

	// Kind: Identifies this object as a list of workers.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *WorkerListResponse) MarshalJSON() ([]byte, error) {
	type noMethod WorkerListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "coordinate.customFieldDef.list":

type CustomFieldDefListCall struct {
	s            *Service
	teamId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of custom field definitions for a team.
func (r *CustomFieldDefService) List(teamId string) *CustomFieldDefListCall {
	c := &CustomFieldDefListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CustomFieldDefListCall) QuotaUser(quotaUser string) *CustomFieldDefListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CustomFieldDefListCall) UserIP(userIP string) *CustomFieldDefListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CustomFieldDefListCall) Fields(s ...googleapi.Field) *CustomFieldDefListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CustomFieldDefListCall) IfNoneMatch(entityTag string) *CustomFieldDefListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CustomFieldDefListCall) Context(ctx context.Context) *CustomFieldDefListCall {
	c.ctx_ = ctx
	return c
}

func (c *CustomFieldDefListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/custom_fields")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
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

// Do executes the "coordinate.customFieldDef.list" call.
// Exactly one of *CustomFieldDefListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *CustomFieldDefListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CustomFieldDefListCall) Do() (*CustomFieldDefListResponse, error) {
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
	ret := &CustomFieldDefListResponse{
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
	//   "path": "teams/{teamId}/custom_fields",
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
	s            *Service
	teamId       string
	jobId        uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves a job, including all the changes made to the job.
func (r *JobsService) Get(teamId string, jobId uint64) *JobsGetCall {
	c := &JobsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.jobId = jobId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *JobsGetCall) QuotaUser(quotaUser string) *JobsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *JobsGetCall) UserIP(userIP string) *JobsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsGetCall) Fields(s ...googleapi.Field) *JobsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *JobsGetCall) IfNoneMatch(entityTag string) *JobsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsGetCall) Context(ctx context.Context) *JobsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
		"jobId":  strconv.FormatUint(c.jobId, 10),
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

// Do executes the "coordinate.jobs.get" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsGetCall) Do() (*Job, error) {
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
	ret := &Job{
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
	//   "path": "teams/{teamId}/jobs/{jobId}",
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
	s          *Service
	teamId     string
	job        *Job
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Inserts a new job. Only the state field of the job should be
// set.
func (r *JobsService) Insert(teamId string, address string, lat float64, lng float64, title string, job *Job) *JobsInsertCall {
	c := &JobsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.urlParams_.Set("address", address)
	c.urlParams_.Set("lat", fmt.Sprint(lat))
	c.urlParams_.Set("lng", fmt.Sprint(lng))
	c.urlParams_.Set("title", title)
	c.job = job
	return c
}

// Assignee sets the optional parameter "assignee": Assignee email
// address, or empty string to unassign.
func (c *JobsInsertCall) Assignee(assignee string) *JobsInsertCall {
	c.urlParams_.Set("assignee", assignee)
	return c
}

// CustomField sets the optional parameter "customField": Sets the value
// of custom fields. To set a custom field, pass the field id (from
// /team/teamId/custom_fields), a URL escaped '=' character, and the
// desired value as a parameter. For example, customField=12%3DAlice.
// Repeat the parameter for each custom field. Note that '=' cannot
// appear in the parameter value. Specifying an invalid, or inactive
// enum field will result in an error 500.
func (c *JobsInsertCall) CustomField(customField ...string) *JobsInsertCall {
	c.urlParams_.SetMulti("customField", append([]string{}, customField...))
	return c
}

// CustomerName sets the optional parameter "customerName": Customer
// name
func (c *JobsInsertCall) CustomerName(customerName string) *JobsInsertCall {
	c.urlParams_.Set("customerName", customerName)
	return c
}

// CustomerPhoneNumber sets the optional parameter
// "customerPhoneNumber": Customer phone number
func (c *JobsInsertCall) CustomerPhoneNumber(customerPhoneNumber string) *JobsInsertCall {
	c.urlParams_.Set("customerPhoneNumber", customerPhoneNumber)
	return c
}

// Note sets the optional parameter "note": Job note as newline (Unix)
// separated string
func (c *JobsInsertCall) Note(note string) *JobsInsertCall {
	c.urlParams_.Set("note", note)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *JobsInsertCall) QuotaUser(quotaUser string) *JobsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *JobsInsertCall) UserIP(userIP string) *JobsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsInsertCall) Fields(s ...googleapi.Field) *JobsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsInsertCall) Context(ctx context.Context) *JobsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "coordinate.jobs.insert" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsInsertCall) Do() (*Job, error) {
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
	ret := &Job{
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
	//       "description": "Sets the value of custom fields. To set a custom field, pass the field id (from /team/teamId/custom_fields), a URL escaped '=' character, and the desired value as a parameter. For example, customField=12%3DAlice. Repeat the parameter for each custom field. Note that '=' cannot appear in the parameter value. Specifying an invalid, or inactive enum field will result in an error 500.",
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
	//   "path": "teams/{teamId}/jobs",
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
	s            *Service
	teamId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves jobs created or modified since the given timestamp.
func (r *JobsService) List(teamId string) *JobsListCall {
	c := &JobsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return in one page.
func (c *JobsListCall) MaxResults(maxResults int64) *JobsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// MinModifiedTimestampMs sets the optional parameter
// "minModifiedTimestampMs": Minimum time a job was modified in
// milliseconds since epoch.
func (c *JobsListCall) MinModifiedTimestampMs(minModifiedTimestampMs uint64) *JobsListCall {
	c.urlParams_.Set("minModifiedTimestampMs", fmt.Sprint(minModifiedTimestampMs))
	return c
}

// OmitJobChanges sets the optional parameter "omitJobChanges": Whether
// to omit detail job history information.
func (c *JobsListCall) OmitJobChanges(omitJobChanges bool) *JobsListCall {
	c.urlParams_.Set("omitJobChanges", fmt.Sprint(omitJobChanges))
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
func (c *JobsListCall) PageToken(pageToken string) *JobsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *JobsListCall) QuotaUser(quotaUser string) *JobsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *JobsListCall) UserIP(userIP string) *JobsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsListCall) Fields(s ...googleapi.Field) *JobsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *JobsListCall) IfNoneMatch(entityTag string) *JobsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsListCall) Context(ctx context.Context) *JobsListCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
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

// Do executes the "coordinate.jobs.list" call.
// Exactly one of *JobListResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *JobListResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *JobsListCall) Do() (*JobListResponse, error) {
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
	ret := &JobListResponse{
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
	//     "omitJobChanges": {
	//       "description": "Whether to omit detail job history information.",
	//       "location": "query",
	//       "type": "boolean"
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
	//   "path": "teams/{teamId}/jobs",
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
	s          *Service
	teamId     string
	jobId      uint64
	job        *Job
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates a job. Fields that are set in the job state will be
// updated. This method supports patch semantics.
func (r *JobsService) Patch(teamId string, jobId uint64, job *Job) *JobsPatchCall {
	c := &JobsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.jobId = jobId
	c.job = job
	return c
}

// Address sets the optional parameter "address": Job address as newline
// (Unix) separated string
func (c *JobsPatchCall) Address(address string) *JobsPatchCall {
	c.urlParams_.Set("address", address)
	return c
}

// Assignee sets the optional parameter "assignee": Assignee email
// address, or empty string to unassign.
func (c *JobsPatchCall) Assignee(assignee string) *JobsPatchCall {
	c.urlParams_.Set("assignee", assignee)
	return c
}

// CustomField sets the optional parameter "customField": Sets the value
// of custom fields. To set a custom field, pass the field id (from
// /team/teamId/custom_fields), a URL escaped '=' character, and the
// desired value as a parameter. For example, customField=12%3DAlice.
// Repeat the parameter for each custom field. Note that '=' cannot
// appear in the parameter value. Specifying an invalid, or inactive
// enum field will result in an error 500.
func (c *JobsPatchCall) CustomField(customField ...string) *JobsPatchCall {
	c.urlParams_.SetMulti("customField", append([]string{}, customField...))
	return c
}

// CustomerName sets the optional parameter "customerName": Customer
// name
func (c *JobsPatchCall) CustomerName(customerName string) *JobsPatchCall {
	c.urlParams_.Set("customerName", customerName)
	return c
}

// CustomerPhoneNumber sets the optional parameter
// "customerPhoneNumber": Customer phone number
func (c *JobsPatchCall) CustomerPhoneNumber(customerPhoneNumber string) *JobsPatchCall {
	c.urlParams_.Set("customerPhoneNumber", customerPhoneNumber)
	return c
}

// Lat sets the optional parameter "lat": The latitude coordinate of
// this job's location.
func (c *JobsPatchCall) Lat(lat float64) *JobsPatchCall {
	c.urlParams_.Set("lat", fmt.Sprint(lat))
	return c
}

// Lng sets the optional parameter "lng": The longitude coordinate of
// this job's location.
func (c *JobsPatchCall) Lng(lng float64) *JobsPatchCall {
	c.urlParams_.Set("lng", fmt.Sprint(lng))
	return c
}

// Note sets the optional parameter "note": Job note as newline (Unix)
// separated string
func (c *JobsPatchCall) Note(note string) *JobsPatchCall {
	c.urlParams_.Set("note", note)
	return c
}

// Progress sets the optional parameter "progress": Job progress
//
// Possible values:
//   "COMPLETED" - Completed
//   "IN_PROGRESS" - In progress
//   "NOT_ACCEPTED" - Not accepted
//   "NOT_STARTED" - Not started
//   "OBSOLETE" - Obsolete
func (c *JobsPatchCall) Progress(progress string) *JobsPatchCall {
	c.urlParams_.Set("progress", progress)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *JobsPatchCall) QuotaUser(quotaUser string) *JobsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Title sets the optional parameter "title": Job title
func (c *JobsPatchCall) Title(title string) *JobsPatchCall {
	c.urlParams_.Set("title", title)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *JobsPatchCall) UserIP(userIP string) *JobsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsPatchCall) Fields(s ...googleapi.Field) *JobsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsPatchCall) Context(ctx context.Context) *JobsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
		"jobId":  strconv.FormatUint(c.jobId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "coordinate.jobs.patch" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsPatchCall) Do() (*Job, error) {
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
	ret := &Job{
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
	//       "description": "Sets the value of custom fields. To set a custom field, pass the field id (from /team/teamId/custom_fields), a URL escaped '=' character, and the desired value as a parameter. For example, customField=12%3DAlice. Repeat the parameter for each custom field. Note that '=' cannot appear in the parameter value. Specifying an invalid, or inactive enum field will result in an error 500.",
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
	//   "path": "teams/{teamId}/jobs/{jobId}",
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
	s          *Service
	teamId     string
	jobId      uint64
	job        *Job
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates a job. Fields that are set in the job state will be
// updated.
func (r *JobsService) Update(teamId string, jobId uint64, job *Job) *JobsUpdateCall {
	c := &JobsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.jobId = jobId
	c.job = job
	return c
}

// Address sets the optional parameter "address": Job address as newline
// (Unix) separated string
func (c *JobsUpdateCall) Address(address string) *JobsUpdateCall {
	c.urlParams_.Set("address", address)
	return c
}

// Assignee sets the optional parameter "assignee": Assignee email
// address, or empty string to unassign.
func (c *JobsUpdateCall) Assignee(assignee string) *JobsUpdateCall {
	c.urlParams_.Set("assignee", assignee)
	return c
}

// CustomField sets the optional parameter "customField": Sets the value
// of custom fields. To set a custom field, pass the field id (from
// /team/teamId/custom_fields), a URL escaped '=' character, and the
// desired value as a parameter. For example, customField=12%3DAlice.
// Repeat the parameter for each custom field. Note that '=' cannot
// appear in the parameter value. Specifying an invalid, or inactive
// enum field will result in an error 500.
func (c *JobsUpdateCall) CustomField(customField ...string) *JobsUpdateCall {
	c.urlParams_.SetMulti("customField", append([]string{}, customField...))
	return c
}

// CustomerName sets the optional parameter "customerName": Customer
// name
func (c *JobsUpdateCall) CustomerName(customerName string) *JobsUpdateCall {
	c.urlParams_.Set("customerName", customerName)
	return c
}

// CustomerPhoneNumber sets the optional parameter
// "customerPhoneNumber": Customer phone number
func (c *JobsUpdateCall) CustomerPhoneNumber(customerPhoneNumber string) *JobsUpdateCall {
	c.urlParams_.Set("customerPhoneNumber", customerPhoneNumber)
	return c
}

// Lat sets the optional parameter "lat": The latitude coordinate of
// this job's location.
func (c *JobsUpdateCall) Lat(lat float64) *JobsUpdateCall {
	c.urlParams_.Set("lat", fmt.Sprint(lat))
	return c
}

// Lng sets the optional parameter "lng": The longitude coordinate of
// this job's location.
func (c *JobsUpdateCall) Lng(lng float64) *JobsUpdateCall {
	c.urlParams_.Set("lng", fmt.Sprint(lng))
	return c
}

// Note sets the optional parameter "note": Job note as newline (Unix)
// separated string
func (c *JobsUpdateCall) Note(note string) *JobsUpdateCall {
	c.urlParams_.Set("note", note)
	return c
}

// Progress sets the optional parameter "progress": Job progress
//
// Possible values:
//   "COMPLETED" - Completed
//   "IN_PROGRESS" - In progress
//   "NOT_ACCEPTED" - Not accepted
//   "NOT_STARTED" - Not started
//   "OBSOLETE" - Obsolete
func (c *JobsUpdateCall) Progress(progress string) *JobsUpdateCall {
	c.urlParams_.Set("progress", progress)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *JobsUpdateCall) QuotaUser(quotaUser string) *JobsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Title sets the optional parameter "title": Job title
func (c *JobsUpdateCall) Title(title string) *JobsUpdateCall {
	c.urlParams_.Set("title", title)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *JobsUpdateCall) UserIP(userIP string) *JobsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsUpdateCall) Fields(s ...googleapi.Field) *JobsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsUpdateCall) Context(ctx context.Context) *JobsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
		"jobId":  strconv.FormatUint(c.jobId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "coordinate.jobs.update" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsUpdateCall) Do() (*Job, error) {
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
	ret := &Job{
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
	//       "description": "Sets the value of custom fields. To set a custom field, pass the field id (from /team/teamId/custom_fields), a URL escaped '=' character, and the desired value as a parameter. For example, customField=12%3DAlice. Repeat the parameter for each custom field. Note that '=' cannot appear in the parameter value. Specifying an invalid, or inactive enum field will result in an error 500.",
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
	//   "path": "teams/{teamId}/jobs/{jobId}",
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
	s            *Service
	teamId       string
	workerEmail  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of locations for a worker.
func (r *LocationService) List(teamId string, workerEmail string, startTimestampMs uint64) *LocationListCall {
	c := &LocationListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.workerEmail = workerEmail
	c.urlParams_.Set("startTimestampMs", fmt.Sprint(startTimestampMs))
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return in one page.
func (c *LocationListCall) MaxResults(maxResults int64) *LocationListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
func (c *LocationListCall) PageToken(pageToken string) *LocationListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *LocationListCall) QuotaUser(quotaUser string) *LocationListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *LocationListCall) UserIP(userIP string) *LocationListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LocationListCall) Fields(s ...googleapi.Field) *LocationListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *LocationListCall) IfNoneMatch(entityTag string) *LocationListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LocationListCall) Context(ctx context.Context) *LocationListCall {
	c.ctx_ = ctx
	return c
}

func (c *LocationListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/workers/{workerEmail}/locations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId":      c.teamId,
		"workerEmail": c.workerEmail,
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

// Do executes the "coordinate.location.list" call.
// Exactly one of *LocationListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LocationListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *LocationListCall) Do() (*LocationListResponse, error) {
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
	ret := &LocationListResponse{
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
	//   "path": "teams/{teamId}/workers/{workerEmail}/locations",
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
	s            *Service
	teamId       string
	jobId        uint64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the schedule for a job.
func (r *ScheduleService) Get(teamId string, jobId uint64) *ScheduleGetCall {
	c := &ScheduleGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.jobId = jobId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ScheduleGetCall) QuotaUser(quotaUser string) *ScheduleGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ScheduleGetCall) UserIP(userIP string) *ScheduleGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ScheduleGetCall) Fields(s ...googleapi.Field) *ScheduleGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ScheduleGetCall) IfNoneMatch(entityTag string) *ScheduleGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ScheduleGetCall) Context(ctx context.Context) *ScheduleGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ScheduleGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs/{jobId}/schedule")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
		"jobId":  strconv.FormatUint(c.jobId, 10),
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

// Do executes the "coordinate.schedule.get" call.
// Exactly one of *Schedule or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Schedule.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ScheduleGetCall) Do() (*Schedule, error) {
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
	ret := &Schedule{
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
	//   "path": "teams/{teamId}/jobs/{jobId}/schedule",
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
	s          *Service
	teamId     string
	jobId      uint64
	schedule   *Schedule
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Replaces the schedule of a job with the provided schedule.
// This method supports patch semantics.
func (r *ScheduleService) Patch(teamId string, jobId uint64, schedule *Schedule) *SchedulePatchCall {
	c := &SchedulePatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.jobId = jobId
	c.schedule = schedule
	return c
}

// AllDay sets the optional parameter "allDay": Whether the job is
// scheduled for the whole day. Time of day in start/end times is
// ignored if this is true.
func (c *SchedulePatchCall) AllDay(allDay bool) *SchedulePatchCall {
	c.urlParams_.Set("allDay", fmt.Sprint(allDay))
	return c
}

// Duration sets the optional parameter "duration": Job duration in
// milliseconds.
func (c *SchedulePatchCall) Duration(duration uint64) *SchedulePatchCall {
	c.urlParams_.Set("duration", fmt.Sprint(duration))
	return c
}

// EndTime sets the optional parameter "endTime": Scheduled end time in
// milliseconds since epoch.
func (c *SchedulePatchCall) EndTime(endTime uint64) *SchedulePatchCall {
	c.urlParams_.Set("endTime", fmt.Sprint(endTime))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *SchedulePatchCall) QuotaUser(quotaUser string) *SchedulePatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// StartTime sets the optional parameter "startTime": Scheduled start
// time in milliseconds since epoch.
func (c *SchedulePatchCall) StartTime(startTime uint64) *SchedulePatchCall {
	c.urlParams_.Set("startTime", fmt.Sprint(startTime))
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *SchedulePatchCall) UserIP(userIP string) *SchedulePatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SchedulePatchCall) Fields(s ...googleapi.Field) *SchedulePatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SchedulePatchCall) Context(ctx context.Context) *SchedulePatchCall {
	c.ctx_ = ctx
	return c
}

func (c *SchedulePatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.schedule)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs/{jobId}/schedule")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
		"jobId":  strconv.FormatUint(c.jobId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "coordinate.schedule.patch" call.
// Exactly one of *Schedule or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Schedule.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SchedulePatchCall) Do() (*Schedule, error) {
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
	ret := &Schedule{
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
	//   "path": "teams/{teamId}/jobs/{jobId}/schedule",
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
	s          *Service
	teamId     string
	jobId      uint64
	schedule   *Schedule
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Replaces the schedule of a job with the provided schedule.
func (r *ScheduleService) Update(teamId string, jobId uint64, schedule *Schedule) *ScheduleUpdateCall {
	c := &ScheduleUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	c.jobId = jobId
	c.schedule = schedule
	return c
}

// AllDay sets the optional parameter "allDay": Whether the job is
// scheduled for the whole day. Time of day in start/end times is
// ignored if this is true.
func (c *ScheduleUpdateCall) AllDay(allDay bool) *ScheduleUpdateCall {
	c.urlParams_.Set("allDay", fmt.Sprint(allDay))
	return c
}

// Duration sets the optional parameter "duration": Job duration in
// milliseconds.
func (c *ScheduleUpdateCall) Duration(duration uint64) *ScheduleUpdateCall {
	c.urlParams_.Set("duration", fmt.Sprint(duration))
	return c
}

// EndTime sets the optional parameter "endTime": Scheduled end time in
// milliseconds since epoch.
func (c *ScheduleUpdateCall) EndTime(endTime uint64) *ScheduleUpdateCall {
	c.urlParams_.Set("endTime", fmt.Sprint(endTime))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ScheduleUpdateCall) QuotaUser(quotaUser string) *ScheduleUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// StartTime sets the optional parameter "startTime": Scheduled start
// time in milliseconds since epoch.
func (c *ScheduleUpdateCall) StartTime(startTime uint64) *ScheduleUpdateCall {
	c.urlParams_.Set("startTime", fmt.Sprint(startTime))
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ScheduleUpdateCall) UserIP(userIP string) *ScheduleUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ScheduleUpdateCall) Fields(s ...googleapi.Field) *ScheduleUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ScheduleUpdateCall) Context(ctx context.Context) *ScheduleUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *ScheduleUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.schedule)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/jobs/{jobId}/schedule")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
		"jobId":  strconv.FormatUint(c.jobId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "coordinate.schedule.update" call.
// Exactly one of *Schedule or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Schedule.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ScheduleUpdateCall) Do() (*Schedule, error) {
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
	ret := &Schedule{
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
	//   "path": "teams/{teamId}/jobs/{jobId}/schedule",
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

// method id "coordinate.team.list":

type TeamListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of teams for a user.
func (r *TeamService) List() *TeamListCall {
	c := &TeamListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Admin sets the optional parameter "admin": Whether to include teams
// for which the user has the Admin role.
func (c *TeamListCall) Admin(admin bool) *TeamListCall {
	c.urlParams_.Set("admin", fmt.Sprint(admin))
	return c
}

// Dispatcher sets the optional parameter "dispatcher": Whether to
// include teams for which the user has the Dispatcher role.
func (c *TeamListCall) Dispatcher(dispatcher bool) *TeamListCall {
	c.urlParams_.Set("dispatcher", fmt.Sprint(dispatcher))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TeamListCall) QuotaUser(quotaUser string) *TeamListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TeamListCall) UserIP(userIP string) *TeamListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Worker sets the optional parameter "worker": Whether to include teams
// for which the user has the Worker role.
func (c *TeamListCall) Worker(worker bool) *TeamListCall {
	c.urlParams_.Set("worker", fmt.Sprint(worker))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TeamListCall) Fields(s ...googleapi.Field) *TeamListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TeamListCall) IfNoneMatch(entityTag string) *TeamListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TeamListCall) Context(ctx context.Context) *TeamListCall {
	c.ctx_ = ctx
	return c
}

func (c *TeamListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams")
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

// Do executes the "coordinate.team.list" call.
// Exactly one of *TeamListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TeamListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TeamListCall) Do() (*TeamListResponse, error) {
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
	ret := &TeamListResponse{
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
	//   "description": "Retrieves a list of teams for a user.",
	//   "httpMethod": "GET",
	//   "id": "coordinate.team.list",
	//   "parameters": {
	//     "admin": {
	//       "description": "Whether to include teams for which the user has the Admin role.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "dispatcher": {
	//       "description": "Whether to include teams for which the user has the Dispatcher role.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "worker": {
	//       "description": "Whether to include teams for which the user has the Worker role.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "teams",
	//   "response": {
	//     "$ref": "TeamListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}

// method id "coordinate.worker.list":

type WorkerListCall struct {
	s            *Service
	teamId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of workers in a team.
func (r *WorkerService) List(teamId string) *WorkerListCall {
	c := &WorkerListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.teamId = teamId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *WorkerListCall) QuotaUser(quotaUser string) *WorkerListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *WorkerListCall) UserIP(userIP string) *WorkerListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *WorkerListCall) Fields(s ...googleapi.Field) *WorkerListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *WorkerListCall) IfNoneMatch(entityTag string) *WorkerListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *WorkerListCall) Context(ctx context.Context) *WorkerListCall {
	c.ctx_ = ctx
	return c
}

func (c *WorkerListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "teams/{teamId}/workers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"teamId": c.teamId,
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

// Do executes the "coordinate.worker.list" call.
// Exactly one of *WorkerListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *WorkerListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *WorkerListCall) Do() (*WorkerListResponse, error) {
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
	ret := &WorkerListResponse{
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
	//   "path": "teams/{teamId}/workers",
	//   "response": {
	//     "$ref": "WorkerListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/coordinate",
	//     "https://www.googleapis.com/auth/coordinate.readonly"
	//   ]
	// }

}
