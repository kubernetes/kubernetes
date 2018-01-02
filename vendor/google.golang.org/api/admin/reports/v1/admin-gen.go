// Package admin provides access to the Admin Reports API.
//
// See https://developers.google.com/admin-sdk/reports/
//
// Usage example:
//
//   import "google.golang.org/api/admin/reports/v1"
//   ...
//   adminService, err := admin.New(oauthHttpClient)
package admin // import "google.golang.org/api/admin/reports/v1"

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

const apiId = "admin:reports_v1"
const apiName = "admin"
const apiVersion = "reports_v1"
const basePath = "https://www.googleapis.com/admin/reports/v1/"

// OAuth2 scopes used by this API.
const (
	// View audit reports for your G Suite domain
	AdminReportsAuditReadonlyScope = "https://www.googleapis.com/auth/admin.reports.audit.readonly"

	// View usage reports for your G Suite domain
	AdminReportsUsageReadonlyScope = "https://www.googleapis.com/auth/admin.reports.usage.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Activities = NewActivitiesService(s)
	s.Channels = NewChannelsService(s)
	s.CustomerUsageReports = NewCustomerUsageReportsService(s)
	s.UserUsageReport = NewUserUsageReportService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Activities *ActivitiesService

	Channels *ChannelsService

	CustomerUsageReports *CustomerUsageReportsService

	UserUsageReport *UserUsageReportService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewActivitiesService(s *Service) *ActivitiesService {
	rs := &ActivitiesService{s: s}
	return rs
}

type ActivitiesService struct {
	s *Service
}

func NewChannelsService(s *Service) *ChannelsService {
	rs := &ChannelsService{s: s}
	return rs
}

type ChannelsService struct {
	s *Service
}

func NewCustomerUsageReportsService(s *Service) *CustomerUsageReportsService {
	rs := &CustomerUsageReportsService{s: s}
	return rs
}

type CustomerUsageReportsService struct {
	s *Service
}

func NewUserUsageReportService(s *Service) *UserUsageReportService {
	rs := &UserUsageReportService{s: s}
	return rs
}

type UserUsageReportService struct {
	s *Service
}

// Activities: JSON template for a collection of activites.
type Activities struct {
	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Items: Each record in read response.
	Items []*Activity `json:"items,omitempty"`

	// Kind: Kind of list response this is.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Token for retrieving the next page
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

func (s *Activities) MarshalJSON() ([]byte, error) {
	type noMethod Activities
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Activity: JSON template for the activity resource.
type Activity struct {
	// Actor: User doing the action.
	Actor *ActivityActor `json:"actor,omitempty"`

	// Etag: ETag of the entry.
	Etag string `json:"etag,omitempty"`

	// Events: Activity events.
	Events []*ActivityEvents `json:"events,omitempty"`

	// Id: Unique identifier for each activity record.
	Id *ActivityId `json:"id,omitempty"`

	// IpAddress: IP Address of the user doing the action.
	IpAddress string `json:"ipAddress,omitempty"`

	// Kind: Kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// OwnerDomain: Domain of source customer.
	OwnerDomain string `json:"ownerDomain,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Actor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Actor") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Activity) MarshalJSON() ([]byte, error) {
	type noMethod Activity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ActivityActor: User doing the action.
type ActivityActor struct {
	// CallerType: User or OAuth 2LO request.
	CallerType string `json:"callerType,omitempty"`

	// Email: Email address of the user.
	Email string `json:"email,omitempty"`

	// Key: For OAuth 2LO API requests, consumer_key of the requestor.
	Key string `json:"key,omitempty"`

	// ProfileId: Obfuscated user id of the user.
	ProfileId string `json:"profileId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CallerType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CallerType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ActivityActor) MarshalJSON() ([]byte, error) {
	type noMethod ActivityActor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ActivityEvents struct {
	// Name: Name of event.
	Name string `json:"name,omitempty"`

	// Parameters: Parameter value pairs for various applications.
	Parameters []*ActivityEventsParameters `json:"parameters,omitempty"`

	// Type: Type of event.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Name") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ActivityEvents) MarshalJSON() ([]byte, error) {
	type noMethod ActivityEvents
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ActivityEventsParameters struct {
	// BoolValue: Boolean value of the parameter.
	BoolValue bool `json:"boolValue,omitempty"`

	// IntValue: Integral value of the parameter.
	IntValue int64 `json:"intValue,omitempty,string"`

	// MultiIntValue: Multi-int value of the parameter.
	MultiIntValue googleapi.Int64s `json:"multiIntValue,omitempty"`

	// MultiValue: Multi-string value of the parameter.
	MultiValue []string `json:"multiValue,omitempty"`

	// Name: The name of the parameter.
	Name string `json:"name,omitempty"`

	// Value: String value of the parameter.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoolValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoolValue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ActivityEventsParameters) MarshalJSON() ([]byte, error) {
	type noMethod ActivityEventsParameters
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ActivityId: Unique identifier for each activity record.
type ActivityId struct {
	// ApplicationName: Application name to which the event belongs.
	ApplicationName string `json:"applicationName,omitempty"`

	// CustomerId: Obfuscated customer ID of the source customer.
	CustomerId string `json:"customerId,omitempty"`

	// Time: Time of occurrence of the activity.
	Time string `json:"time,omitempty"`

	// UniqueQualifier: Unique qualifier if multiple events have the same
	// time.
	UniqueQualifier int64 `json:"uniqueQualifier,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "ApplicationName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApplicationName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ActivityId) MarshalJSON() ([]byte, error) {
	type noMethod ActivityId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Channel: An notification channel used to watch for resource changes.
type Channel struct {
	// Address: The address where notifications are delivered for this
	// channel.
	Address string `json:"address,omitempty"`

	// Expiration: Date and time of notification channel expiration,
	// expressed as a Unix timestamp, in milliseconds. Optional.
	Expiration int64 `json:"expiration,omitempty,string"`

	// Id: A UUID or similar unique string that identifies this channel.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a notification channel used to watch for
	// changes to a resource. Value: the fixed string "api#channel".
	Kind string `json:"kind,omitempty"`

	// Params: Additional parameters controlling delivery channel behavior.
	// Optional.
	Params map[string]string `json:"params,omitempty"`

	// Payload: A Boolean value to indicate whether payload is wanted.
	// Optional.
	Payload bool `json:"payload,omitempty"`

	// ResourceId: An opaque ID that identifies the resource being watched
	// on this channel. Stable across different API versions.
	ResourceId string `json:"resourceId,omitempty"`

	// ResourceUri: A version-specific identifier for the watched resource.
	ResourceUri string `json:"resourceUri,omitempty"`

	// Token: An arbitrary string delivered to the target address with each
	// notification delivered over this channel. Optional.
	Token string `json:"token,omitempty"`

	// Type: The type of delivery mechanism used for this channel.
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Address") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Address") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Channel) MarshalJSON() ([]byte, error) {
	type noMethod Channel
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UsageReport: JSON template for a usage report.
type UsageReport struct {
	// Date: The date to which the record belongs.
	Date string `json:"date,omitempty"`

	// Entity: Information about the type of the item.
	Entity *UsageReportEntity `json:"entity,omitempty"`

	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Kind: The kind of object.
	Kind string `json:"kind,omitempty"`

	// Parameters: Parameter value pairs for various applications.
	Parameters []*UsageReportParameters `json:"parameters,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Date") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Date") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UsageReport) MarshalJSON() ([]byte, error) {
	type noMethod UsageReport
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UsageReportEntity: Information about the type of the item.
type UsageReportEntity struct {
	// CustomerId: Obfuscated customer id for the record.
	CustomerId string `json:"customerId,omitempty"`

	// ProfileId: Obfuscated user id for the record.
	ProfileId string `json:"profileId,omitempty"`

	// Type: The type of item, can be a customer or user.
	Type string `json:"type,omitempty"`

	// UserEmail: user's email.
	UserEmail string `json:"userEmail,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CustomerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CustomerId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UsageReportEntity) MarshalJSON() ([]byte, error) {
	type noMethod UsageReportEntity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type UsageReportParameters struct {
	// BoolValue: Boolean value of the parameter.
	BoolValue bool `json:"boolValue,omitempty"`

	// DatetimeValue: RFC 3339 formatted value of the parameter.
	DatetimeValue string `json:"datetimeValue,omitempty"`

	// IntValue: Integral value of the parameter.
	IntValue int64 `json:"intValue,omitempty,string"`

	// MsgValue: Nested message value of the parameter.
	MsgValue []googleapi.RawMessage `json:"msgValue,omitempty"`

	// Name: The name of the parameter.
	Name string `json:"name,omitempty"`

	// StringValue: String value of the parameter.
	StringValue string `json:"stringValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoolValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoolValue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UsageReportParameters) MarshalJSON() ([]byte, error) {
	type noMethod UsageReportParameters
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UsageReports: JSON template for a collection of usage reports.
type UsageReports struct {
	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Kind: The kind of object.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Token for retrieving the next page
	NextPageToken string `json:"nextPageToken,omitempty"`

	// UsageReports: Various application parameter records.
	UsageReports []*UsageReport `json:"usageReports,omitempty"`

	// Warnings: Warnings if any.
	Warnings []*UsageReportsWarnings `json:"warnings,omitempty"`

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

func (s *UsageReports) MarshalJSON() ([]byte, error) {
	type noMethod UsageReports
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type UsageReportsWarnings struct {
	// Code: Machine readable code / warning type.
	Code string `json:"code,omitempty"`

	// Data: Key-Value pairs to give detailed information on the warning.
	Data []*UsageReportsWarningsData `json:"data,omitempty"`

	// Message: Human readable message for the warning.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UsageReportsWarnings) MarshalJSON() ([]byte, error) {
	type noMethod UsageReportsWarnings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type UsageReportsWarningsData struct {
	// Key: Key associated with a key-value pair to give detailed
	// information on the warning.
	Key string `json:"key,omitempty"`

	// Value: Value associated with a key-value pair to give detailed
	// information on the warning.
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

func (s *UsageReportsWarningsData) MarshalJSON() ([]byte, error) {
	type noMethod UsageReportsWarningsData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "reports.activities.list":

type ActivitiesListCall struct {
	s               *Service
	userKey         string
	applicationName string
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// List: Retrieves a list of activities for a specific customer and
// application.
func (r *ActivitiesService) List(userKey string, applicationName string) *ActivitiesListCall {
	c := &ActivitiesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userKey = userKey
	c.applicationName = applicationName
	return c
}

// ActorIpAddress sets the optional parameter "actorIpAddress": IP
// Address of host where the event was performed. Supports both IPv4 and
// IPv6 addresses.
func (c *ActivitiesListCall) ActorIpAddress(actorIpAddress string) *ActivitiesListCall {
	c.urlParams_.Set("actorIpAddress", actorIpAddress)
	return c
}

// CustomerId sets the optional parameter "customerId": Represents the
// customer for which the data is to be fetched.
func (c *ActivitiesListCall) CustomerId(customerId string) *ActivitiesListCall {
	c.urlParams_.Set("customerId", customerId)
	return c
}

// EndTime sets the optional parameter "endTime": Return events which
// occurred at or before this time.
func (c *ActivitiesListCall) EndTime(endTime string) *ActivitiesListCall {
	c.urlParams_.Set("endTime", endTime)
	return c
}

// EventName sets the optional parameter "eventName": Name of the event
// being queried.
func (c *ActivitiesListCall) EventName(eventName string) *ActivitiesListCall {
	c.urlParams_.Set("eventName", eventName)
	return c
}

// Filters sets the optional parameter "filters": Event parameters in
// the form [parameter1 name][operator][parameter1 value],[parameter2
// name][operator][parameter2 value],...
func (c *ActivitiesListCall) Filters(filters string) *ActivitiesListCall {
	c.urlParams_.Set("filters", filters)
	return c
}

// MaxResults sets the optional parameter "maxResults": Number of
// activity records to be shown in each page.
func (c *ActivitiesListCall) MaxResults(maxResults int64) *ActivitiesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token to specify
// next page.
func (c *ActivitiesListCall) PageToken(pageToken string) *ActivitiesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// StartTime sets the optional parameter "startTime": Return events
// which occurred at or after this time.
func (c *ActivitiesListCall) StartTime(startTime string) *ActivitiesListCall {
	c.urlParams_.Set("startTime", startTime)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ActivitiesListCall) Fields(s ...googleapi.Field) *ActivitiesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ActivitiesListCall) IfNoneMatch(entityTag string) *ActivitiesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ActivitiesListCall) Context(ctx context.Context) *ActivitiesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ActivitiesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ActivitiesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "activity/users/{userKey}/applications/{applicationName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userKey":         c.userKey,
		"applicationName": c.applicationName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "reports.activities.list" call.
// Exactly one of *Activities or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Activities.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ActivitiesListCall) Do(opts ...googleapi.CallOption) (*Activities, error) {
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
	ret := &Activities{
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
	//   "description": "Retrieves a list of activities for a specific customer and application.",
	//   "httpMethod": "GET",
	//   "id": "reports.activities.list",
	//   "parameterOrder": [
	//     "userKey",
	//     "applicationName"
	//   ],
	//   "parameters": {
	//     "actorIpAddress": {
	//       "description": "IP Address of host where the event was performed. Supports both IPv4 and IPv6 addresses.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "applicationName": {
	//       "description": "Application name for which the events are to be retrieved.",
	//       "location": "path",
	//       "pattern": "(admin)|(calendar)|(drive)|(login)|(mobile)|(token)|(groups)|(saml)|(chat)|(gplus)|(rules)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customerId": {
	//       "description": "Represents the customer for which the data is to be fetched.",
	//       "location": "query",
	//       "pattern": "C.+",
	//       "type": "string"
	//     },
	//     "endTime": {
	//       "description": "Return events which occurred at or before this time.",
	//       "location": "query",
	//       "pattern": "(\\d\\d\\d\\d)-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):(\\d\\d)(?:\\.(\\d+))?(?:(Z)|([-+])(\\d\\d):(\\d\\d))",
	//       "type": "string"
	//     },
	//     "eventName": {
	//       "description": "Name of the event being queried.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "filters": {
	//       "description": "Event parameters in the form [parameter1 name][operator][parameter1 value],[parameter2 name][operator][parameter2 value],...",
	//       "location": "query",
	//       "pattern": "(.+[\u003c,\u003c=,==,\u003e=,\u003e,\u003c\u003e].+,)*(.+[\u003c,\u003c=,==,\u003e=,\u003e,\u003c\u003e].+)",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Number of activity records to be shown in each page.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Token to specify next page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startTime": {
	//       "description": "Return events which occurred at or after this time.",
	//       "location": "query",
	//       "pattern": "(\\d\\d\\d\\d)-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):(\\d\\d)(?:\\.(\\d+))?(?:(Z)|([-+])(\\d\\d):(\\d\\d))",
	//       "type": "string"
	//     },
	//     "userKey": {
	//       "description": "Represents the profile id or the user email for which the data should be filtered. When 'all' is specified as the userKey, it returns usageReports for all users.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activity/users/{userKey}/applications/{applicationName}",
	//   "response": {
	//     "$ref": "Activities"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/admin.reports.audit.readonly"
	//   ],
	//   "supportsSubscription": true
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ActivitiesListCall) Pages(ctx context.Context, f func(*Activities) error) error {
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

// method id "reports.activities.watch":

type ActivitiesWatchCall struct {
	s               *Service
	userKey         string
	applicationName string
	channel         *Channel
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Watch: Push changes to activities
func (r *ActivitiesService) Watch(userKey string, applicationName string, channel *Channel) *ActivitiesWatchCall {
	c := &ActivitiesWatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userKey = userKey
	c.applicationName = applicationName
	c.channel = channel
	return c
}

// ActorIpAddress sets the optional parameter "actorIpAddress": IP
// Address of host where the event was performed. Supports both IPv4 and
// IPv6 addresses.
func (c *ActivitiesWatchCall) ActorIpAddress(actorIpAddress string) *ActivitiesWatchCall {
	c.urlParams_.Set("actorIpAddress", actorIpAddress)
	return c
}

// CustomerId sets the optional parameter "customerId": Represents the
// customer for which the data is to be fetched.
func (c *ActivitiesWatchCall) CustomerId(customerId string) *ActivitiesWatchCall {
	c.urlParams_.Set("customerId", customerId)
	return c
}

// EndTime sets the optional parameter "endTime": Return events which
// occurred at or before this time.
func (c *ActivitiesWatchCall) EndTime(endTime string) *ActivitiesWatchCall {
	c.urlParams_.Set("endTime", endTime)
	return c
}

// EventName sets the optional parameter "eventName": Name of the event
// being queried.
func (c *ActivitiesWatchCall) EventName(eventName string) *ActivitiesWatchCall {
	c.urlParams_.Set("eventName", eventName)
	return c
}

// Filters sets the optional parameter "filters": Event parameters in
// the form [parameter1 name][operator][parameter1 value],[parameter2
// name][operator][parameter2 value],...
func (c *ActivitiesWatchCall) Filters(filters string) *ActivitiesWatchCall {
	c.urlParams_.Set("filters", filters)
	return c
}

// MaxResults sets the optional parameter "maxResults": Number of
// activity records to be shown in each page.
func (c *ActivitiesWatchCall) MaxResults(maxResults int64) *ActivitiesWatchCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token to specify
// next page.
func (c *ActivitiesWatchCall) PageToken(pageToken string) *ActivitiesWatchCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// StartTime sets the optional parameter "startTime": Return events
// which occurred at or after this time.
func (c *ActivitiesWatchCall) StartTime(startTime string) *ActivitiesWatchCall {
	c.urlParams_.Set("startTime", startTime)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ActivitiesWatchCall) Fields(s ...googleapi.Field) *ActivitiesWatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ActivitiesWatchCall) Context(ctx context.Context) *ActivitiesWatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ActivitiesWatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ActivitiesWatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "activity/users/{userKey}/applications/{applicationName}/watch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userKey":         c.userKey,
		"applicationName": c.applicationName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "reports.activities.watch" call.
// Exactly one of *Channel or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Channel.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ActivitiesWatchCall) Do(opts ...googleapi.CallOption) (*Channel, error) {
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
	ret := &Channel{
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
	//   "description": "Push changes to activities",
	//   "httpMethod": "POST",
	//   "id": "reports.activities.watch",
	//   "parameterOrder": [
	//     "userKey",
	//     "applicationName"
	//   ],
	//   "parameters": {
	//     "actorIpAddress": {
	//       "description": "IP Address of host where the event was performed. Supports both IPv4 and IPv6 addresses.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "applicationName": {
	//       "description": "Application name for which the events are to be retrieved.",
	//       "location": "path",
	//       "pattern": "(admin)|(calendar)|(drive)|(login)|(mobile)|(token)|(groups)|(saml)|(chat)|(gplus)|(rules)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customerId": {
	//       "description": "Represents the customer for which the data is to be fetched.",
	//       "location": "query",
	//       "pattern": "C.+",
	//       "type": "string"
	//     },
	//     "endTime": {
	//       "description": "Return events which occurred at or before this time.",
	//       "location": "query",
	//       "pattern": "(\\d\\d\\d\\d)-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):(\\d\\d)(?:\\.(\\d+))?(?:(Z)|([-+])(\\d\\d):(\\d\\d))",
	//       "type": "string"
	//     },
	//     "eventName": {
	//       "description": "Name of the event being queried.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "filters": {
	//       "description": "Event parameters in the form [parameter1 name][operator][parameter1 value],[parameter2 name][operator][parameter2 value],...",
	//       "location": "query",
	//       "pattern": "(.+[\u003c,\u003c=,==,\u003e=,\u003e,\u003c\u003e].+,)*(.+[\u003c,\u003c=,==,\u003e=,\u003e,\u003c\u003e].+)",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Number of activity records to be shown in each page.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Token to specify next page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startTime": {
	//       "description": "Return events which occurred at or after this time.",
	//       "location": "query",
	//       "pattern": "(\\d\\d\\d\\d)-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):(\\d\\d)(?:\\.(\\d+))?(?:(Z)|([-+])(\\d\\d):(\\d\\d))",
	//       "type": "string"
	//     },
	//     "userKey": {
	//       "description": "Represents the profile id or the user email for which the data should be filtered. When 'all' is specified as the userKey, it returns usageReports for all users.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activity/users/{userKey}/applications/{applicationName}/watch",
	//   "request": {
	//     "$ref": "Channel",
	//     "parameterName": "resource"
	//   },
	//   "response": {
	//     "$ref": "Channel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/admin.reports.audit.readonly"
	//   ],
	//   "supportsSubscription": true
	// }

}

// method id "admin.channels.stop":

type ChannelsStopCall struct {
	s          *Service
	channel    *Channel
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Stop: Stop watching resources through this channel
func (r *ChannelsService) Stop(channel *Channel) *ChannelsStopCall {
	c := &ChannelsStopCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.channel = channel
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ChannelsStopCall) Fields(s ...googleapi.Field) *ChannelsStopCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ChannelsStopCall) Context(ctx context.Context) *ChannelsStopCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ChannelsStopCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ChannelsStopCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "/admin/reports_v1/channels/stop")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "admin.channels.stop" call.
func (c *ChannelsStopCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Stop watching resources through this channel",
	//   "httpMethod": "POST",
	//   "id": "admin.channels.stop",
	//   "path": "/admin/reports_v1/channels/stop",
	//   "request": {
	//     "$ref": "Channel",
	//     "parameterName": "resource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/admin.reports.audit.readonly"
	//   ]
	// }

}

// method id "reports.customerUsageReports.get":

type CustomerUsageReportsGetCall struct {
	s            *Service
	date         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a report which is a collection of properties /
// statistics for a specific customer.
func (r *CustomerUsageReportsService) Get(date string) *CustomerUsageReportsGetCall {
	c := &CustomerUsageReportsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.date = date
	return c
}

// CustomerId sets the optional parameter "customerId": Represents the
// customer for which the data is to be fetched.
func (c *CustomerUsageReportsGetCall) CustomerId(customerId string) *CustomerUsageReportsGetCall {
	c.urlParams_.Set("customerId", customerId)
	return c
}

// PageToken sets the optional parameter "pageToken": Token to specify
// next page.
func (c *CustomerUsageReportsGetCall) PageToken(pageToken string) *CustomerUsageReportsGetCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Parameters sets the optional parameter "parameters": Represents the
// application name, parameter name pairs to fetch in csv as
// app_name1:param_name1, app_name2:param_name2.
func (c *CustomerUsageReportsGetCall) Parameters(parameters string) *CustomerUsageReportsGetCall {
	c.urlParams_.Set("parameters", parameters)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CustomerUsageReportsGetCall) Fields(s ...googleapi.Field) *CustomerUsageReportsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CustomerUsageReportsGetCall) IfNoneMatch(entityTag string) *CustomerUsageReportsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CustomerUsageReportsGetCall) Context(ctx context.Context) *CustomerUsageReportsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CustomerUsageReportsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CustomerUsageReportsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "usage/dates/{date}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"date": c.date,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "reports.customerUsageReports.get" call.
// Exactly one of *UsageReports or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UsageReports.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CustomerUsageReportsGetCall) Do(opts ...googleapi.CallOption) (*UsageReports, error) {
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
	ret := &UsageReports{
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
	//   "description": "Retrieves a report which is a collection of properties / statistics for a specific customer.",
	//   "httpMethod": "GET",
	//   "id": "reports.customerUsageReports.get",
	//   "parameterOrder": [
	//     "date"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Represents the customer for which the data is to be fetched.",
	//       "location": "query",
	//       "pattern": "C.+",
	//       "type": "string"
	//     },
	//     "date": {
	//       "description": "Represents the date in yyyy-mm-dd format for which the data is to be fetched.",
	//       "location": "path",
	//       "pattern": "(\\d){4}-(\\d){2}-(\\d){2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Token to specify next page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parameters": {
	//       "description": "Represents the application name, parameter name pairs to fetch in csv as app_name1:param_name1, app_name2:param_name2.",
	//       "location": "query",
	//       "pattern": "(((accounts)|(app_maker)|(apps_scripts)|(classroom)|(cros)|(gmail)|(calendar)|(docs)|(gplus)|(sites)|(device_management)|(drive)):[^,]+,)*(((accounts)|(app_maker)|(apps_scripts)|(classroom)|(cros)|(gmail)|(calendar)|(docs)|(gplus)|(sites)|(device_management)|(drive)):[^,]+)",
	//       "type": "string"
	//     }
	//   },
	//   "path": "usage/dates/{date}",
	//   "response": {
	//     "$ref": "UsageReports"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/admin.reports.usage.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *CustomerUsageReportsGetCall) Pages(ctx context.Context, f func(*UsageReports) error) error {
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

// method id "reports.userUsageReport.get":

type UserUsageReportGetCall struct {
	s            *Service
	userKey      string
	date         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a report which is a collection of properties /
// statistics for a set of users.
func (r *UserUsageReportService) Get(userKey string, date string) *UserUsageReportGetCall {
	c := &UserUsageReportGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userKey = userKey
	c.date = date
	return c
}

// CustomerId sets the optional parameter "customerId": Represents the
// customer for which the data is to be fetched.
func (c *UserUsageReportGetCall) CustomerId(customerId string) *UserUsageReportGetCall {
	c.urlParams_.Set("customerId", customerId)
	return c
}

// Filters sets the optional parameter "filters": Represents the set of
// filters including parameter operator value.
func (c *UserUsageReportGetCall) Filters(filters string) *UserUsageReportGetCall {
	c.urlParams_.Set("filters", filters)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return. Maximum allowed is 1000
func (c *UserUsageReportGetCall) MaxResults(maxResults int64) *UserUsageReportGetCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token to specify
// next page.
func (c *UserUsageReportGetCall) PageToken(pageToken string) *UserUsageReportGetCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Parameters sets the optional parameter "parameters": Represents the
// application name, parameter name pairs to fetch in csv as
// app_name1:param_name1, app_name2:param_name2.
func (c *UserUsageReportGetCall) Parameters(parameters string) *UserUsageReportGetCall {
	c.urlParams_.Set("parameters", parameters)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserUsageReportGetCall) Fields(s ...googleapi.Field) *UserUsageReportGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UserUsageReportGetCall) IfNoneMatch(entityTag string) *UserUsageReportGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserUsageReportGetCall) Context(ctx context.Context) *UserUsageReportGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UserUsageReportGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UserUsageReportGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "usage/users/{userKey}/dates/{date}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userKey": c.userKey,
		"date":    c.date,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "reports.userUsageReport.get" call.
// Exactly one of *UsageReports or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UsageReports.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UserUsageReportGetCall) Do(opts ...googleapi.CallOption) (*UsageReports, error) {
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
	ret := &UsageReports{
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
	//   "description": "Retrieves a report which is a collection of properties / statistics for a set of users.",
	//   "httpMethod": "GET",
	//   "id": "reports.userUsageReport.get",
	//   "parameterOrder": [
	//     "userKey",
	//     "date"
	//   ],
	//   "parameters": {
	//     "customerId": {
	//       "description": "Represents the customer for which the data is to be fetched.",
	//       "location": "query",
	//       "pattern": "C.+",
	//       "type": "string"
	//     },
	//     "date": {
	//       "description": "Represents the date in yyyy-mm-dd format for which the data is to be fetched.",
	//       "location": "path",
	//       "pattern": "(\\d){4}-(\\d){2}-(\\d){2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filters": {
	//       "description": "Represents the set of filters including parameter operator value.",
	//       "location": "query",
	//       "pattern": "(((accounts)|(classroom)|(cros)|(gmail)|(calendar)|(docs)|(gplus)|(sites)|(device_management)|(drive)):[a-z0-9_]+[\u003c,\u003c=,==,\u003e=,\u003e,!=][^,]+,)*(((accounts)|(classroom)|(cros)|(gmail)|(calendar)|(docs)|(gplus)|(sites)|(device_management)|(drive)):[a-z0-9_]+[\u003c,\u003c=,==,\u003e=,\u003e,!=][^,]+)",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return. Maximum allowed is 1000",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Token to specify next page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parameters": {
	//       "description": "Represents the application name, parameter name pairs to fetch in csv as app_name1:param_name1, app_name2:param_name2.",
	//       "location": "query",
	//       "pattern": "(((accounts)|(classroom)|(cros)|(gmail)|(calendar)|(docs)|(gplus)|(sites)|(device_management)|(drive)):[^,]+,)*(((accounts)|(classroom)|(cros)|(gmail)|(calendar)|(docs)|(gplus)|(sites)|(device_management)|(drive)):[^,]+)",
	//       "type": "string"
	//     },
	//     "userKey": {
	//       "description": "Represents the profile id or the user email for which the data should be filtered.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "usage/users/{userKey}/dates/{date}",
	//   "response": {
	//     "$ref": "UsageReports"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/admin.reports.usage.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *UserUsageReportGetCall) Pages(ctx context.Context, f func(*UsageReports) error) error {
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
