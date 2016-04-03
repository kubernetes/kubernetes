// Package dfareporting provides access to the DFA Reporting API.
//
// See https://developers.google.com/doubleclick-advertisers/reporting/
//
// Usage example:
//
//   import "google.golang.org/api/dfareporting/v1"
//   ...
//   dfareportingService, err := dfareporting.New(oauthHttpClient)
package dfareporting // import "google.golang.org/api/dfareporting/v1"

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

const apiId = "dfareporting:v1"
const apiName = "dfareporting"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/dfareporting/v1/"

// OAuth2 scopes used by this API.
const (
	// View and manage DoubleClick for Advertisers reports
	DfareportingScope = "https://www.googleapis.com/auth/dfareporting"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.DimensionValues = NewDimensionValuesService(s)
	s.Files = NewFilesService(s)
	s.Reports = NewReportsService(s)
	s.UserProfiles = NewUserProfilesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	DimensionValues *DimensionValuesService

	Files *FilesService

	Reports *ReportsService

	UserProfiles *UserProfilesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewDimensionValuesService(s *Service) *DimensionValuesService {
	rs := &DimensionValuesService{s: s}
	return rs
}

type DimensionValuesService struct {
	s *Service
}

func NewFilesService(s *Service) *FilesService {
	rs := &FilesService{s: s}
	return rs
}

type FilesService struct {
	s *Service
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	rs.Files = NewReportsFilesService(s)
	return rs
}

type ReportsService struct {
	s *Service

	Files *ReportsFilesService
}

func NewReportsFilesService(s *Service) *ReportsFilesService {
	rs := &ReportsFilesService{s: s}
	return rs
}

type ReportsFilesService struct {
	s *Service
}

func NewUserProfilesService(s *Service) *UserProfilesService {
	rs := &UserProfilesService{s: s}
	return rs
}

type UserProfilesService struct {
	s *Service
}

// DimensionFilter: Represents a dimension filter.
type DimensionFilter struct {
	// DimensionName: The name of the dimension to filter.
	DimensionName string `json:"dimensionName,omitempty"`

	// Kind: Kind of resource this is, in this case
	// dfareporting#dimensionFilter.
	Kind string `json:"kind,omitempty"`

	// Value: The value of the dimension to filter for.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DimensionName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DimensionFilter) MarshalJSON() ([]byte, error) {
	type noMethod DimensionFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DimensionValue: Represents a DimensionValue resource.
type DimensionValue struct {
	// DimensionName: Name of the dimension.
	DimensionName string `json:"dimensionName,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Id: The ID associated with the value if available.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case
	// dfareporting#dimensionValue.
	Kind string `json:"kind,omitempty"`

	// Value: The value of the dimension.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DimensionName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DimensionValue) MarshalJSON() ([]byte, error) {
	type noMethod DimensionValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DimensionValueList: Represents the list of DimensionValue resources.
type DimensionValueList struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The dimension values returned in this response.
	Items []*DimensionValue `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case
	// dfareporting#dimensionValueList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through dimension
	// values. To retrieve the next page of results, set the next request's
	// "pageToken" to the value of this field. The page token is only valid
	// for a limited amount of time and should not be persisted.
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
}

func (s *DimensionValueList) MarshalJSON() ([]byte, error) {
	type noMethod DimensionValueList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DimensionValueRequest: Represents a DimensionValuesRequest.
type DimensionValueRequest struct {
	// DimensionName: The name of the dimension values should be requested
	// for.
	DimensionName string `json:"dimensionName,omitempty"`

	// EndDate: The end date of the date range for which to retrieve
	// dimension values. A string of the format: "yyyy-MM-dd".
	EndDate string `json:"endDate,omitempty"`

	// Filters: List of filters to filter values by. The filters are ANDed.
	Filters []*DimensionFilter `json:"filters,omitempty"`

	// Kind: Kind of request this is, in this case
	// dfareporting#dimensionValueRequest.
	Kind string `json:"kind,omitempty"`

	// StartDate: The start date of the date range for which to retrieve
	// dimension values. A string of the format: "yyyy-MM-dd".
	StartDate string `json:"startDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DimensionName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DimensionValueRequest) MarshalJSON() ([]byte, error) {
	type noMethod DimensionValueRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// File: Represents a File resource. A File contains the meta-data for a
// report run. It shows the status of the run and holds the urls to the
// generated report data if the run is finished and the status is
// "REPORT_AVAILABLE".
type File struct {
	// DateRange: The date range for which the file has report data.
	DateRange *FileDateRange `json:"dateRange,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// FileName: The file name of the file.
	FileName string `json:"fileName,omitempty"`

	// Id: The unique ID of this report file.
	Id int64 `json:"id,omitempty,string"`

	// Kind: Kind of resource this is, in this case dfareporting#file.
	Kind string `json:"kind,omitempty"`

	// LastModifiedTime: The timestamp in milliseconds since epoch when this
	// file was last modified.
	LastModifiedTime int64 `json:"lastModifiedTime,omitempty,string"`

	// ReportId: The ID of the report this file was generated from.
	ReportId int64 `json:"reportId,omitempty,string"`

	// Status: The status of the report file, one of:
	// - "PROCESSING"
	// - "REPORT_AVAILABLE"
	// - "FAILED"
	// - "CANCELLED"
	Status string `json:"status,omitempty"`

	// Urls: The urls where the completed report file can be downloaded.
	Urls *FileUrls `json:"urls,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DateRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *File) MarshalJSON() ([]byte, error) {
	type noMethod File
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// FileDateRange: The date range for which the file has report data.
type FileDateRange struct {
	// EndDate: The end date of the date range, inclusive. A string of the
	// format: "yyyy-MM-dd".
	EndDate string `json:"endDate,omitempty"`

	// StartDate: The start date of the date range, inclusive. A string of
	// the format: "yyyy-MM-dd".
	StartDate string `json:"startDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *FileDateRange) MarshalJSON() ([]byte, error) {
	type noMethod FileDateRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// FileUrls: The urls where the completed report file can be downloaded.
type FileUrls struct {
	// Csv: Urls for generated CSV data.
	Csv *FileUrlsCsv `json:"csv,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Csv") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *FileUrls) MarshalJSON() ([]byte, error) {
	type noMethod FileUrls
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// FileUrlsCsv: Urls for generated CSV data.
type FileUrlsCsv struct {
	// ApiUrl: The url for downloading the report data through the API.
	ApiUrl string `json:"apiUrl,omitempty"`

	// BrowserUrl: The url for downloading the report data through a
	// browser.
	BrowserUrl string `json:"browserUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApiUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *FileUrlsCsv) MarshalJSON() ([]byte, error) {
	type noMethod FileUrlsCsv
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// FileList: Represents the list of File resources.
type FileList struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The files returned in this response.
	Items []*File `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case dfareporting#fileList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through files. To
	// retrieve the next page of results, set the next request's "pageToken"
	// to the value of this field. The page token is only valid for a
	// limited amount of time and should not be persisted.
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
}

func (s *FileList) MarshalJSON() ([]byte, error) {
	type noMethod FileList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Report: Represents a Report resource.
type Report struct {
	// AccountId: The account id this report belongs to.
	AccountId int64 `json:"accountId,omitempty,string"`

	// Criteria: The report criteria.
	Criteria *ReportCriteria `json:"criteria,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// FileName: The file name used when generating report files for this
	// report.
	FileName string `json:"fileName,omitempty"`

	// Id: The unique ID identifying this report resource.
	Id int64 `json:"id,omitempty,string"`

	// Kind: Kind of resource this is, in this case dfareporting#report.
	Kind string `json:"kind,omitempty"`

	// LastModifiedTime: The timestamp (in milliseconds since epoch) of when
	// this report was last modified.
	LastModifiedTime uint64 `json:"lastModifiedTime,omitempty,string"`

	// Name: The name of the report.
	Name string `json:"name,omitempty"`

	// OwnerProfileId: The user profile id of the owner of this report.
	OwnerProfileId int64 `json:"ownerProfileId,omitempty,string"`

	// Schedule: The report's schedule. Can only be set if the report's
	// 'dateRange' is a relative date range and the relative date range is
	// not "TODAY".
	Schedule *ReportSchedule `json:"schedule,omitempty"`

	// SubAccountId: The subbaccount id this report belongs to if
	// applicable.
	SubAccountId int64 `json:"subAccountId,omitempty,string"`

	// Type: The type of the report, currently only "STANDARD" is supported.
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Report) MarshalJSON() ([]byte, error) {
	type noMethod Report
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportCriteria: The report criteria.
type ReportCriteria struct {
	// Activities: Activity group.
	Activities *ReportCriteriaActivities `json:"activities,omitempty"`

	// CustomRichMediaEvents: Custom Rich Media Events group.
	CustomRichMediaEvents *ReportCriteriaCustomRichMediaEvents `json:"customRichMediaEvents,omitempty"`

	// DateRange: The date range this report should be run for.
	DateRange *ReportCriteriaDateRange `json:"dateRange,omitempty"`

	// DimensionFilters: The list of filters dimensions are filtered
	// on.
	// Filters for different dimensions are ANDed, filters for the same
	// dimension are grouped together and ORed.
	DimensionFilters []*DimensionValue `json:"dimensionFilters,omitempty"`

	// Dimensions: The list of dimensions the report should include.
	Dimensions []*SortedDimension `json:"dimensions,omitempty"`

	// MetricNames: The list of names of metrics the report should include.
	MetricNames []string `json:"metricNames,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Activities") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportCriteria) MarshalJSON() ([]byte, error) {
	type noMethod ReportCriteria
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportCriteriaActivities: Activity group.
type ReportCriteriaActivities struct {
	// Filters: List of activity filters. The dimension values need to be
	// all either of type "dfa:activity" or "dfa:activityGroup".
	Filters []*DimensionValue `json:"filters,omitempty"`

	// MetricNames: List of names of floodlight activity metrics.
	MetricNames []string `json:"metricNames,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportCriteriaActivities) MarshalJSON() ([]byte, error) {
	type noMethod ReportCriteriaActivities
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportCriteriaCustomRichMediaEvents: Custom Rich Media Events group.
type ReportCriteriaCustomRichMediaEvents struct {
	// FilteredEventIds: List of custom rich media event IDs. Dimension
	// values must be all of type dfa:richMediaEventTypeIdAndName.
	FilteredEventIds []*DimensionValue `json:"filteredEventIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FilteredEventIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportCriteriaCustomRichMediaEvents) MarshalJSON() ([]byte, error) {
	type noMethod ReportCriteriaCustomRichMediaEvents
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportCriteriaDateRange: The date range this report should be run
// for.
type ReportCriteriaDateRange struct {
	// EndDate: The end date of the date range, inclusive. A string of the
	// format: "yyyy-MM-dd".
	EndDate string `json:"endDate,omitempty"`

	// RelativeDateRange: The date range relative to the date of when the
	// report is run, one of:
	// - "TODAY"
	// - "YESTERDAY"
	// - "WEEK_TO_DATE"
	// - "MONTH_TO_DATE"
	// - "QUARTER_TO_DATE"
	// - "YEAR_TO_DATE"
	// - "PREVIOUS_WEEK"
	// - "PREVIOUS_MONTH"
	// - "PREVIOUS_QUARTER"
	// - "PREVIOUS_YEAR"
	// - "LAST_7_DAYS"
	// - "LAST_30_DAYS"
	// - "LAST_90_DAYS"
	// - "LAST_365_DAYS"
	// - "LAST_24_MONTHS"
	RelativeDateRange string `json:"relativeDateRange,omitempty"`

	// StartDate: The start date of the date range, inclusive. A string of
	// the format: "yyyy-MM-dd".
	StartDate string `json:"startDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportCriteriaDateRange) MarshalJSON() ([]byte, error) {
	type noMethod ReportCriteriaDateRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportSchedule: The report's schedule. Can only be set if the
// report's 'dateRange' is a relative date range and the relative date
// range is not "TODAY".
type ReportSchedule struct {
	// Active: Whether the schedule is active or not. Must be set to either
	// true or false.
	Active bool `json:"active,omitempty"`

	// Every: Defines every how many days, weeks or months the report should
	// be run. Needs to be set when "repeats" is either "DAILY", "WEEKLY" or
	// "MONTHLY".
	Every int64 `json:"every,omitempty"`

	// ExpirationDate: The expiration date when the scheduled report stops
	// running.
	ExpirationDate string `json:"expirationDate,omitempty"`

	// Repeats: The interval the report is repeated for, one of:
	// - "DAILY", also requires field "every" to be set.
	// - "WEEKLY", also requires fields "every" and "repeatsOnWeekDays" to
	// be set.
	// - "TWICE_A_MONTH"
	// - "MONTHLY", also requires fields "every" and "runsOnDayOfMonth" to
	// be set.
	// - "QUARTERLY"
	// - "YEARLY"
	Repeats string `json:"repeats,omitempty"`

	// RepeatsOnWeekDays: List of week days "WEEKLY" scheduled reports
	// should run on.
	RepeatsOnWeekDays []string `json:"repeatsOnWeekDays,omitempty"`

	// RunsOnDayOfMonth: Enum to define for "MONTHLY" scheduled reports
	// whether reports should be repeated on the same day of the month as
	// "startDate" or the same day of the week of the month. Possible values
	// are:
	// - DAY_OF_MONTH
	// - WEEK_OF_MONTH
	// Example: If 'startDate' is Monday, April 2nd 2012 (2012-04-02),
	// "DAY_OF_MONTH" would run subsequent reports on the 2nd of every
	// Month, and "WEEK_OF_MONTH" would run subsequent reports on the first
	// Monday of the month.
	RunsOnDayOfMonth string `json:"runsOnDayOfMonth,omitempty"`

	// StartDate: Start date of date range for which scheduled reports
	// should be run.
	StartDate string `json:"startDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Active") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportSchedule) MarshalJSON() ([]byte, error) {
	type noMethod ReportSchedule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportList: Represents the list of reports.
type ReportList struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The reports returned in this response.
	Items []*Report `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case dfareporting#reportList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through reports. To
	// retrieve the next page of results, set the next request's "pageToken"
	// to the value of this field. The page token is only valid for a
	// limited amount of time and should not be persisted.
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
}

func (s *ReportList) MarshalJSON() ([]byte, error) {
	type noMethod ReportList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SortedDimension: Represents a sorted dimension.
type SortedDimension struct {
	// Kind: Kind of resource this is, in this case
	// dfareporting#sortedDimension.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the dimension.
	Name string `json:"name,omitempty"`

	// SortOrder: An optional sort order for the dimension column, one of:
	//
	// - "ASCENDING"
	// - "DESCENDING"
	SortOrder string `json:"sortOrder,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SortedDimension) MarshalJSON() ([]byte, error) {
	type noMethod SortedDimension
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UserProfile: Represents a UserProfile resource.
type UserProfile struct {
	// AccountId: The account ID this profile belongs to.
	AccountId int64 `json:"accountId,omitempty,string"`

	// AccountName: The account name this profile belongs to.
	AccountName string `json:"accountName,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Kind: Kind of resource this is, in this case
	// dfareporting#userProfile.
	Kind string `json:"kind,omitempty"`

	// ProfileId: The unique ID of the user profile.
	ProfileId int64 `json:"profileId,omitempty,string"`

	// SubAccountId: The sub account ID this profile belongs to if
	// applicable.
	SubAccountId int64 `json:"subAccountId,omitempty,string"`

	// SubAccountName: The sub account name this profile belongs to if
	// applicable.
	SubAccountName string `json:"subAccountName,omitempty"`

	// UserName: The user name.
	UserName string `json:"userName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UserProfile) MarshalJSON() ([]byte, error) {
	type noMethod UserProfile
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UserProfileList: Represents the list of user profiles.
type UserProfileList struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The user profiles returned in this response.
	Items []*UserProfile `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case
	// dfareporting#userProfileList.
	Kind string `json:"kind,omitempty"`

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
}

func (s *UserProfileList) MarshalJSON() ([]byte, error) {
	type noMethod UserProfileList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "dfareporting.dimensionValues.query":

type DimensionValuesQueryCall struct {
	s                     *Service
	profileId             int64
	dimensionvaluerequest *DimensionValueRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
}

// Query: Retrieves list of report dimension values for a list of
// filters.
func (r *DimensionValuesService) Query(profileId int64, dimensionvaluerequest *DimensionValueRequest) *DimensionValuesQueryCall {
	c := &DimensionValuesQueryCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.dimensionvaluerequest = dimensionvaluerequest
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *DimensionValuesQueryCall) MaxResults(maxResults int64) *DimensionValuesQueryCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *DimensionValuesQueryCall) PageToken(pageToken string) *DimensionValuesQueryCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DimensionValuesQueryCall) QuotaUser(quotaUser string) *DimensionValuesQueryCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DimensionValuesQueryCall) UserIP(userIP string) *DimensionValuesQueryCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DimensionValuesQueryCall) Fields(s ...googleapi.Field) *DimensionValuesQueryCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DimensionValuesQueryCall) Context(ctx context.Context) *DimensionValuesQueryCall {
	c.ctx_ = ctx
	return c
}

func (c *DimensionValuesQueryCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dimensionvaluerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/dimensionvalues/query")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dfareporting.dimensionValues.query" call.
// Exactly one of *DimensionValueList or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DimensionValueList.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DimensionValuesQueryCall) Do() (*DimensionValueList, error) {
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
	ret := &DimensionValueList{
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
	//   "description": "Retrieves list of report dimension values for a list of filters.",
	//   "httpMethod": "POST",
	//   "id": "dfareporting.dimensionValues.query",
	//   "parameterOrder": [
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous result page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "The DFA user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/dimensionvalues/query",
	//   "request": {
	//     "$ref": "DimensionValueRequest"
	//   },
	//   "response": {
	//     "$ref": "DimensionValueList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.files.list":

type FilesListCall struct {
	s            *Service
	profileId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists files for a user profile.
func (r *FilesService) List(profileId int64) *FilesListCall {
	c := &FilesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *FilesListCall) MaxResults(maxResults int64) *FilesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *FilesListCall) PageToken(pageToken string) *FilesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *FilesListCall) QuotaUser(quotaUser string) *FilesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// SortField sets the optional parameter "sortField": The field to sort
// the list by.
//
// Possible values:
//   "ID" - Sort by file id.
//   "LAST_MODIFIED_TIME" (default) - Sort by 'lastmodifiedAt' field.
func (c *FilesListCall) SortField(sortField string) *FilesListCall {
	c.urlParams_.Set("sortField", sortField)
	return c
}

// SortOrder sets the optional parameter "sortOrder": Order of sorted
// results, default is 'DESCENDING'.
//
// Possible values:
//   "ASCENDING" - Ascending order.
//   "DESCENDING" (default) - Descending order.
func (c *FilesListCall) SortOrder(sortOrder string) *FilesListCall {
	c.urlParams_.Set("sortOrder", sortOrder)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *FilesListCall) UserIP(userIP string) *FilesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FilesListCall) Fields(s ...googleapi.Field) *FilesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FilesListCall) IfNoneMatch(entityTag string) *FilesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FilesListCall) Context(ctx context.Context) *FilesListCall {
	c.ctx_ = ctx
	return c
}

func (c *FilesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/files")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
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

// Do executes the "dfareporting.files.list" call.
// Exactly one of *FileList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *FileList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *FilesListCall) Do() (*FileList, error) {
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
	ret := &FileList{
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
	//   "description": "Lists files for a user profile.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.files.list",
	//   "parameterOrder": [
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous result page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "The DFA profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sortField": {
	//       "default": "LAST_MODIFIED_TIME",
	//       "description": "The field to sort the list by.",
	//       "enum": [
	//         "ID",
	//         "LAST_MODIFIED_TIME"
	//       ],
	//       "enumDescriptions": [
	//         "Sort by file id.",
	//         "Sort by 'lastmodifiedAt' field."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sortOrder": {
	//       "default": "DESCENDING",
	//       "description": "Order of sorted results, default is 'DESCENDING'.",
	//       "enum": [
	//         "ASCENDING",
	//         "DESCENDING"
	//       ],
	//       "enumDescriptions": [
	//         "Ascending order.",
	//         "Descending order."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/files",
	//   "response": {
	//     "$ref": "FileList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.delete":

type ReportsDeleteCall struct {
	s          *Service
	profileId  int64
	reportId   int64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a report by its id.
func (r *ReportsService) Delete(profileId int64, reportId int64) *ReportsDeleteCall {
	c := &ReportsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsDeleteCall) QuotaUser(quotaUser string) *ReportsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsDeleteCall) UserIP(userIP string) *ReportsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsDeleteCall) Fields(s ...googleapi.Field) *ReportsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsDeleteCall) Context(ctx context.Context) *ReportsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
		"reportId":  strconv.FormatInt(c.reportId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dfareporting.reports.delete" call.
func (c *ReportsDeleteCall) Do() error {
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
	//   "description": "Deletes a report by its id.",
	//   "httpMethod": "DELETE",
	//   "id": "dfareporting.reports.delete",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The id of the report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/{reportId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.get":

type ReportsGetCall struct {
	s            *Service
	profileId    int64
	reportId     int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves a report by its id.
func (r *ReportsService) Get(profileId int64, reportId int64) *ReportsGetCall {
	c := &ReportsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsGetCall) QuotaUser(quotaUser string) *ReportsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsGetCall) UserIP(userIP string) *ReportsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsGetCall) Fields(s ...googleapi.Field) *ReportsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReportsGetCall) IfNoneMatch(entityTag string) *ReportsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsGetCall) Context(ctx context.Context) *ReportsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
		"reportId":  strconv.FormatInt(c.reportId, 10),
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

// Do executes the "dfareporting.reports.get" call.
// Exactly one of *Report or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Report.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ReportsGetCall) Do() (*Report, error) {
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
	ret := &Report{
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
	//   "description": "Retrieves a report by its id.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.reports.get",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The id of the report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/{reportId}",
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.insert":

type ReportsInsertCall struct {
	s          *Service
	profileId  int64
	report     *Report
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Creates a report.
func (r *ReportsService) Insert(profileId int64, report *Report) *ReportsInsertCall {
	c := &ReportsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.report = report
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsInsertCall) QuotaUser(quotaUser string) *ReportsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsInsertCall) UserIP(userIP string) *ReportsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsInsertCall) Fields(s ...googleapi.Field) *ReportsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsInsertCall) Context(ctx context.Context) *ReportsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.report)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dfareporting.reports.insert" call.
// Exactly one of *Report or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Report.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ReportsInsertCall) Do() (*Report, error) {
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
	ret := &Report{
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
	//   "description": "Creates a report.",
	//   "httpMethod": "POST",
	//   "id": "dfareporting.reports.insert",
	//   "parameterOrder": [
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports",
	//   "request": {
	//     "$ref": "Report"
	//   },
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.list":

type ReportsListCall struct {
	s            *Service
	profileId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves list of reports.
func (r *ReportsService) List(profileId int64) *ReportsListCall {
	c := &ReportsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *ReportsListCall) MaxResults(maxResults int64) *ReportsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *ReportsListCall) PageToken(pageToken string) *ReportsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsListCall) QuotaUser(quotaUser string) *ReportsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// SortField sets the optional parameter "sortField": The field to sort
// the list by.
//
// Possible values:
//   "ID" - Sort by report id.
//   "LAST_MODIFIED_TIME" (default) - Sort by 'lastModifiedTime' field.
//   "NAME" - Sort by display name of reports.
func (c *ReportsListCall) SortField(sortField string) *ReportsListCall {
	c.urlParams_.Set("sortField", sortField)
	return c
}

// SortOrder sets the optional parameter "sortOrder": Order of sorted
// results, default is 'DESCENDING'.
//
// Possible values:
//   "ASCENDING" - Ascending order.
//   "DESCENDING" (default) - Descending order.
func (c *ReportsListCall) SortOrder(sortOrder string) *ReportsListCall {
	c.urlParams_.Set("sortOrder", sortOrder)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsListCall) UserIP(userIP string) *ReportsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsListCall) Fields(s ...googleapi.Field) *ReportsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReportsListCall) IfNoneMatch(entityTag string) *ReportsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsListCall) Context(ctx context.Context) *ReportsListCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
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

// Do executes the "dfareporting.reports.list" call.
// Exactly one of *ReportList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ReportList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReportsListCall) Do() (*ReportList, error) {
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
	ret := &ReportList{
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
	//   "description": "Retrieves list of reports.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.reports.list",
	//   "parameterOrder": [
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous result page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "The DFA user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sortField": {
	//       "default": "LAST_MODIFIED_TIME",
	//       "description": "The field to sort the list by.",
	//       "enum": [
	//         "ID",
	//         "LAST_MODIFIED_TIME",
	//         "NAME"
	//       ],
	//       "enumDescriptions": [
	//         "Sort by report id.",
	//         "Sort by 'lastModifiedTime' field.",
	//         "Sort by display name of reports."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sortOrder": {
	//       "default": "DESCENDING",
	//       "description": "Order of sorted results, default is 'DESCENDING'.",
	//       "enum": [
	//         "ASCENDING",
	//         "DESCENDING"
	//       ],
	//       "enumDescriptions": [
	//         "Ascending order.",
	//         "Descending order."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports",
	//   "response": {
	//     "$ref": "ReportList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.patch":

type ReportsPatchCall struct {
	s          *Service
	profileId  int64
	reportId   int64
	report     *Report
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates a report. This method supports patch semantics.
func (r *ReportsService) Patch(profileId int64, reportId int64, report *Report) *ReportsPatchCall {
	c := &ReportsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.reportId = reportId
	c.report = report
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsPatchCall) QuotaUser(quotaUser string) *ReportsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsPatchCall) UserIP(userIP string) *ReportsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsPatchCall) Fields(s ...googleapi.Field) *ReportsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsPatchCall) Context(ctx context.Context) *ReportsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.report)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
		"reportId":  strconv.FormatInt(c.reportId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dfareporting.reports.patch" call.
// Exactly one of *Report or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Report.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ReportsPatchCall) Do() (*Report, error) {
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
	ret := &Report{
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
	//   "description": "Updates a report. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "dfareporting.reports.patch",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The id of the report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/{reportId}",
	//   "request": {
	//     "$ref": "Report"
	//   },
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.run":

type ReportsRunCall struct {
	s          *Service
	profileId  int64
	reportId   int64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Run: Runs a report.
func (r *ReportsService) Run(profileId int64, reportId int64) *ReportsRunCall {
	c := &ReportsRunCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsRunCall) QuotaUser(quotaUser string) *ReportsRunCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Synchronous sets the optional parameter "synchronous": If set and
// true, tries to run the report synchronously.
func (c *ReportsRunCall) Synchronous(synchronous bool) *ReportsRunCall {
	c.urlParams_.Set("synchronous", fmt.Sprint(synchronous))
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsRunCall) UserIP(userIP string) *ReportsRunCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsRunCall) Fields(s ...googleapi.Field) *ReportsRunCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsRunCall) Context(ctx context.Context) *ReportsRunCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsRunCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}/run")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
		"reportId":  strconv.FormatInt(c.reportId, 10),
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dfareporting.reports.run" call.
// Exactly one of *File or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *File.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ReportsRunCall) Do() (*File, error) {
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
	ret := &File{
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
	//   "description": "Runs a report.",
	//   "httpMethod": "POST",
	//   "id": "dfareporting.reports.run",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The id of the report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "synchronous": {
	//       "description": "If set and true, tries to run the report synchronously.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/{reportId}/run",
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.update":

type ReportsUpdateCall struct {
	s          *Service
	profileId  int64
	reportId   int64
	report     *Report
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates a report.
func (r *ReportsService) Update(profileId int64, reportId int64, report *Report) *ReportsUpdateCall {
	c := &ReportsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.reportId = reportId
	c.report = report
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsUpdateCall) QuotaUser(quotaUser string) *ReportsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsUpdateCall) UserIP(userIP string) *ReportsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsUpdateCall) Fields(s ...googleapi.Field) *ReportsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsUpdateCall) Context(ctx context.Context) *ReportsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.report)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
		"reportId":  strconv.FormatInt(c.reportId, 10),
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dfareporting.reports.update" call.
// Exactly one of *Report or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Report.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ReportsUpdateCall) Do() (*Report, error) {
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
	ret := &Report{
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
	//   "description": "Updates a report.",
	//   "httpMethod": "PUT",
	//   "id": "dfareporting.reports.update",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The id of the report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/{reportId}",
	//   "request": {
	//     "$ref": "Report"
	//   },
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.files.get":

type ReportsFilesGetCall struct {
	s            *Service
	profileId    int64
	reportId     int64
	fileId       int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves a report file.
func (r *ReportsFilesService) Get(profileId int64, reportId int64, fileId int64) *ReportsFilesGetCall {
	c := &ReportsFilesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.reportId = reportId
	c.fileId = fileId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsFilesGetCall) QuotaUser(quotaUser string) *ReportsFilesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsFilesGetCall) UserIP(userIP string) *ReportsFilesGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsFilesGetCall) Fields(s ...googleapi.Field) *ReportsFilesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReportsFilesGetCall) IfNoneMatch(entityTag string) *ReportsFilesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsFilesGetCall) Context(ctx context.Context) *ReportsFilesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsFilesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}/files/{fileId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
		"reportId":  strconv.FormatInt(c.reportId, 10),
		"fileId":    strconv.FormatInt(c.fileId, 10),
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

// Do executes the "dfareporting.reports.files.get" call.
// Exactly one of *File or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *File.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ReportsFilesGetCall) Do() (*File, error) {
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
	ret := &File{
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
	//   "description": "Retrieves a report file.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.reports.files.get",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId",
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The id of the report file.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "The DFA profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The id of the report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/{reportId}/files/{fileId}",
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.files.list":

type ReportsFilesListCall struct {
	s            *Service
	profileId    int64
	reportId     int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists files for a report.
func (r *ReportsFilesService) List(profileId int64, reportId int64) *ReportsFilesListCall {
	c := &ReportsFilesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *ReportsFilesListCall) MaxResults(maxResults int64) *ReportsFilesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *ReportsFilesListCall) PageToken(pageToken string) *ReportsFilesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsFilesListCall) QuotaUser(quotaUser string) *ReportsFilesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// SortField sets the optional parameter "sortField": The field to sort
// the list by.
//
// Possible values:
//   "ID" - Sort by file id.
//   "LAST_MODIFIED_TIME" (default) - Sort by 'lastmodifiedAt' field.
func (c *ReportsFilesListCall) SortField(sortField string) *ReportsFilesListCall {
	c.urlParams_.Set("sortField", sortField)
	return c
}

// SortOrder sets the optional parameter "sortOrder": Order of sorted
// results, default is 'DESCENDING'.
//
// Possible values:
//   "ASCENDING" - Ascending order.
//   "DESCENDING" (default) - Descending order.
func (c *ReportsFilesListCall) SortOrder(sortOrder string) *ReportsFilesListCall {
	c.urlParams_.Set("sortOrder", sortOrder)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsFilesListCall) UserIP(userIP string) *ReportsFilesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsFilesListCall) Fields(s ...googleapi.Field) *ReportsFilesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReportsFilesListCall) IfNoneMatch(entityTag string) *ReportsFilesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsFilesListCall) Context(ctx context.Context) *ReportsFilesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsFilesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}/files")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
		"reportId":  strconv.FormatInt(c.reportId, 10),
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

// Do executes the "dfareporting.reports.files.list" call.
// Exactly one of *FileList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *FileList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReportsFilesListCall) Do() (*FileList, error) {
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
	ret := &FileList{
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
	//   "description": "Lists files for a report.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.reports.files.list",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous result page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "The DFA profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The id of the parent report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sortField": {
	//       "default": "LAST_MODIFIED_TIME",
	//       "description": "The field to sort the list by.",
	//       "enum": [
	//         "ID",
	//         "LAST_MODIFIED_TIME"
	//       ],
	//       "enumDescriptions": [
	//         "Sort by file id.",
	//         "Sort by 'lastmodifiedAt' field."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sortOrder": {
	//       "default": "DESCENDING",
	//       "description": "Order of sorted results, default is 'DESCENDING'.",
	//       "enum": [
	//         "ASCENDING",
	//         "DESCENDING"
	//       ],
	//       "enumDescriptions": [
	//         "Ascending order.",
	//         "Descending order."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/{reportId}/files",
	//   "response": {
	//     "$ref": "FileList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.userProfiles.get":

type UserProfilesGetCall struct {
	s            *Service
	profileId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets one user profile by id.
func (r *UserProfilesService) Get(profileId int64) *UserProfilesGetCall {
	c := &UserProfilesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.profileId = profileId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UserProfilesGetCall) QuotaUser(quotaUser string) *UserProfilesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UserProfilesGetCall) UserIP(userIP string) *UserProfilesGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserProfilesGetCall) Fields(s ...googleapi.Field) *UserProfilesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UserProfilesGetCall) IfNoneMatch(entityTag string) *UserProfilesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserProfilesGetCall) Context(ctx context.Context) *UserProfilesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *UserProfilesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"profileId": strconv.FormatInt(c.profileId, 10),
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

// Do executes the "dfareporting.userProfiles.get" call.
// Exactly one of *UserProfile or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserProfile.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UserProfilesGetCall) Do() (*UserProfile, error) {
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
	ret := &UserProfile{
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
	//   "description": "Gets one user profile by id.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.userProfiles.get",
	//   "parameterOrder": [
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The user profile id.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}",
	//   "response": {
	//     "$ref": "UserProfile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.userProfiles.list":

type UserProfilesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves list of user profiles for a user.
func (r *UserProfilesService) List() *UserProfilesListCall {
	c := &UserProfilesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UserProfilesListCall) QuotaUser(quotaUser string) *UserProfilesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UserProfilesListCall) UserIP(userIP string) *UserProfilesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserProfilesListCall) Fields(s ...googleapi.Field) *UserProfilesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UserProfilesListCall) IfNoneMatch(entityTag string) *UserProfilesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserProfilesListCall) Context(ctx context.Context) *UserProfilesListCall {
	c.ctx_ = ctx
	return c
}

func (c *UserProfilesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles")
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

// Do executes the "dfareporting.userProfiles.list" call.
// Exactly one of *UserProfileList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserProfileList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UserProfilesListCall) Do() (*UserProfileList, error) {
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
	ret := &UserProfileList{
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
	//   "description": "Retrieves list of user profiles for a user.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.userProfiles.list",
	//   "path": "userprofiles",
	//   "response": {
	//     "$ref": "UserProfileList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}
