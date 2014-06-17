// Package dfareporting provides access to the DFA Reporting API.
//
// See https://developers.google.com/doubleclick-advertisers/reporting/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/dfareporting/v1.3"
//   ...
//   dfareportingService, err := dfareporting.New(oauthHttpClient)
package dfareporting

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

const apiId = "dfareporting:v1.3"
const apiName = "dfareporting"
const apiVersion = "v1.3"
const basePath = "https://www.googleapis.com/dfareporting/v1.3/"

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
	client   *http.Client
	BasePath string // API endpoint base URL

	DimensionValues *DimensionValuesService

	Files *FilesService

	Reports *ReportsService

	UserProfiles *UserProfilesService
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
	rs.CompatibleFields = NewReportsCompatibleFieldsService(s)
	rs.Files = NewReportsFilesService(s)
	return rs
}

type ReportsService struct {
	s *Service

	CompatibleFields *ReportsCompatibleFieldsService

	Files *ReportsFilesService
}

func NewReportsCompatibleFieldsService(s *Service) *ReportsCompatibleFieldsService {
	rs := &ReportsCompatibleFieldsService{s: s}
	return rs
}

type ReportsCompatibleFieldsService struct {
	s *Service
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

type Activities struct {
	// Filters: List of activity filters. The dimension values need to be
	// all either of type "dfa:activity" or "dfa:activityGroup".
	Filters []*DimensionValue `json:"filters,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#activities.
	Kind string `json:"kind,omitempty"`

	// MetricNames: List of names of floodlight activity metrics.
	MetricNames []string `json:"metricNames,omitempty"`
}

type CompatibleFields struct {
	// CrossDimensionReachReportCompatibleFields: Contains items that are
	// compatible to be selected for a report of type
	// "CROSS_DIMENSION_REACH".
	CrossDimensionReachReportCompatibleFields *CrossDimensionReachReportCompatibleFields `json:"crossDimensionReachReportCompatibleFields,omitempty"`

	// FloodlightReportCompatibleFields: Contains items that are compatible
	// to be selected for a report of type "FLOODLIGHT".
	FloodlightReportCompatibleFields *FloodlightReportCompatibleFields `json:"floodlightReportCompatibleFields,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#compatibleFields.
	Kind string `json:"kind,omitempty"`

	// PathToConversionReportCompatibleFields: Contains items that are
	// compatible to be selected for a report of type "PATH_TO_CONVERSION".
	PathToConversionReportCompatibleFields *PathToConversionReportCompatibleFields `json:"pathToConversionReportCompatibleFields,omitempty"`

	// ReachReportCompatibleFields: Contains items that are compatible to be
	// selected for a report of type "REACH".
	ReachReportCompatibleFields *ReachReportCompatibleFields `json:"reachReportCompatibleFields,omitempty"`

	// ReportCompatibleFields: Contains items that are compatible to be
	// selected for a report of type "STANDARD".
	ReportCompatibleFields *ReportCompatibleFields `json:"reportCompatibleFields,omitempty"`
}

type CrossDimensionReachReportCompatibleFields struct {
	// Breakdown: Dimensions which are compatible to be selected in the
	// "breakdown" section of the report.
	Breakdown []*Dimension `json:"breakdown,omitempty"`

	// DimensionFilters: Dimensions which are compatible to be selected in
	// the "dimensionFilters" section of the report.
	DimensionFilters []*Dimension `json:"dimensionFilters,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#crossDimensionReachReportCompatibleFields.
	Kind string `json:"kind,omitempty"`

	// Metrics: Metrics which are compatible to be selected in the
	// "metricNames" section of the report.
	Metrics []*Metric `json:"metrics,omitempty"`

	// OverlapMetrics: Metrics which are compatible to be selected in the
	// "overlapMetricNames" section of the report.
	OverlapMetrics []*Metric `json:"overlapMetrics,omitempty"`
}

type CustomRichMediaEvents struct {
	// FilteredEventIds: List of custom rich media event IDs. Dimension
	// values must be all of type dfa:richMediaEventTypeIdAndName.
	FilteredEventIds []*DimensionValue `json:"filteredEventIds,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#customRichMediaEvents.
	Kind string `json:"kind,omitempty"`
}

type DateRange struct {
	// EndDate: The end date of the date range, inclusive. A string of the
	// format: "yyyy-MM-dd".
	EndDate string `json:"endDate,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#dateRange.
	Kind string `json:"kind,omitempty"`

	// RelativeDateRange: The date range relative to the date of when the
	// report is run, one of:
	// - "TODAY"
	// - "YESTERDAY"
	// - "WEEK_TO_DATE"
	//
	// - "MONTH_TO_DATE"
	// - "QUARTER_TO_DATE"
	// - "YEAR_TO_DATE"
	// -
	// "PREVIOUS_WEEK"
	// - "PREVIOUS_MONTH"
	// - "PREVIOUS_QUARTER"
	// -
	// "PREVIOUS_YEAR"
	// - "LAST_7_DAYS"
	// - "LAST_30_DAYS"
	// - "LAST_90_DAYS"
	//
	// - "LAST_365_DAYS"
	// - "LAST_24_MONTHS"
	RelativeDateRange string `json:"relativeDateRange,omitempty"`

	// StartDate: The start date of the date range, inclusive. A string of
	// the format: "yyyy-MM-dd".
	StartDate string `json:"startDate,omitempty"`
}

type Dimension struct {
	// Kind: The kind of resource this is, in this case
	// dfareporting#dimension.
	Kind string `json:"kind,omitempty"`

	// Name: The dimension name, e.g. dfa:advertiser
	Name string `json:"name,omitempty"`
}

type DimensionFilter struct {
	// DimensionName: The name of the dimension to filter.
	DimensionName string `json:"dimensionName,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#dimensionFilter.
	Kind string `json:"kind,omitempty"`

	// Value: The value of the dimension to filter.
	Value string `json:"value,omitempty"`
}

type DimensionValue struct {
	// DimensionName: The name of the dimension.
	DimensionName string `json:"dimensionName,omitempty"`

	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Id: The ID associated with the value if available.
	Id string `json:"id,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#dimensionValue.
	Kind string `json:"kind,omitempty"`

	// MatchType: Determines how the 'value' field is matched when
	// filtering. One of:
	// - EXACT (default if not specified)
	// - CONTAINS
	//
	// - BEGINS_WITH
	// - WILDCARD_EXPRESSION (allowing '*' as a placeholder
	// for variable length character sequences, it can be escaped with a
	// backslash.)  Note, only paid search dimensions ('dfa:paidSearch*')
	// allow a matchType other than EXACT.
	MatchType string `json:"matchType,omitempty"`

	// Value: The value of the dimension.
	Value string `json:"value,omitempty"`
}

type DimensionValueList struct {
	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The dimension values returned in this response.
	Items []*DimensionValue `json:"items,omitempty"`

	// Kind: The kind of list this is, in this case
	// dfareporting#dimensionValueList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through dimension
	// values. To retrieve the next page of results, set the next request's
	// "pageToken" to the value of this field. The page token is only valid
	// for a limited amount of time and should not be persisted.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type DimensionValueRequest struct {
	// DimensionName: The name of the dimension for which values should be
	// requested.
	DimensionName string `json:"dimensionName,omitempty"`

	// EndDate: The end date of the date range for which to retrieve
	// dimension values. A string of the format: "yyyy-MM-dd".
	EndDate string `json:"endDate,omitempty"`

	// Filters: The list of filters by which to filter values. The filters
	// are ANDed.
	Filters []*DimensionFilter `json:"filters,omitempty"`

	// Kind: The kind of request this is, in this case
	// dfareporting#dimensionValueRequest.
	Kind string `json:"kind,omitempty"`

	// StartDate: The start date of the date range for which to retrieve
	// dimension values. A string of the format: "yyyy-MM-dd".
	StartDate string `json:"startDate,omitempty"`
}

type File struct {
	// DateRange: The date range for which the file has report data. The
	// date range will always be the absolute date range for which the
	// report is run.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// FileName: The file name of the file.
	FileName string `json:"fileName,omitempty"`

	// Format: The output format of the report. Only available once the file
	// is available.
	Format string `json:"format,omitempty"`

	// Id: The unique ID of this report file.
	Id int64 `json:"id,omitempty,string"`

	// Kind: The kind of resource this is, in this case dfareporting#file.
	Kind string `json:"kind,omitempty"`

	// LastModifiedTime: The timestamp in milliseconds since epoch when this
	// file was last modified.
	LastModifiedTime int64 `json:"lastModifiedTime,omitempty,string"`

	// ReportId: The ID of the report this file was generated from.
	ReportId int64 `json:"reportId,omitempty,string"`

	// Status: The status of the report file, one of:
	// - "PROCESSING"
	// -
	// "REPORT_AVAILABLE"
	// - "FAILED"
	// - "CANCELLED"
	Status string `json:"status,omitempty"`

	// Urls: The urls where the completed report file can be downloaded.
	Urls *FileUrls `json:"urls,omitempty"`
}

type FileUrls struct {
	// ApiUrl: The url for downloading the report data through the API.
	ApiUrl string `json:"apiUrl,omitempty"`

	// BrowserUrl: The url for downloading the report data through a
	// browser.
	BrowserUrl string `json:"browserUrl,omitempty"`
}

type FileList struct {
	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The files returned in this response.
	Items []*File `json:"items,omitempty"`

	// Kind: The kind of list this is, in this case dfareporting#fileList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through files. To
	// retrieve the next page of results, set the next request's "pageToken"
	// to the value of this field. The page token is only valid for a
	// limited amount of time and should not be persisted.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type FloodlightReportCompatibleFields struct {
	// DimensionFilters: Dimensions which are compatible to be selected in
	// the "dimensionFilters" section of the report.
	DimensionFilters []*Dimension `json:"dimensionFilters,omitempty"`

	// Dimensions: Dimensions which are compatible to be selected in the
	// "dimensions" section of the report.
	Dimensions []*Dimension `json:"dimensions,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#floodlightReportCompatibleFields.
	Kind string `json:"kind,omitempty"`

	// Metrics: Metrics which are compatible to be selected in the
	// "metricNames" section of the report.
	Metrics []*Metric `json:"metrics,omitempty"`
}

type Metric struct {
	// Kind: The kind of resource this is, in this case dfareporting#metric.
	Kind string `json:"kind,omitempty"`

	// Name: The metric name, e.g. dfa:impressions
	Name string `json:"name,omitempty"`
}

type PathToConversionReportCompatibleFields struct {
	// ConversionDimensions: Conversion dimensions which are compatible to
	// be selected in the "conversionDimensions" section of the report.
	ConversionDimensions []*Dimension `json:"conversionDimensions,omitempty"`

	// CustomFloodlightVariables: Custom floodlight variables which are
	// compatible to be selected in the "customFloodlightVariables" section
	// of the report.
	CustomFloodlightVariables []*Dimension `json:"customFloodlightVariables,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#pathToConversionReportCompatibleFields.
	Kind string `json:"kind,omitempty"`

	// Metrics: Metrics which are compatible to be selected in the
	// "metricNames" section of the report.
	Metrics []*Metric `json:"metrics,omitempty"`

	// PerInteractionDimensions: Per-interaction dimensions which are
	// compatible to be selected in the "perInteractionDimensions" section
	// of the report.
	PerInteractionDimensions []*Dimension `json:"perInteractionDimensions,omitempty"`
}

type ReachReportCompatibleFields struct {
	// DimensionFilters: Dimensions which are compatible to be selected in
	// the "dimensionFilters" section of the report.
	DimensionFilters []*Dimension `json:"dimensionFilters,omitempty"`

	// Dimensions: Dimensions which are compatible to be selected in the
	// "dimensions" section of the report.
	Dimensions []*Dimension `json:"dimensions,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#reachReportCompatibleFields.
	Kind string `json:"kind,omitempty"`

	// Metrics: Metrics which are compatible to be selected in the
	// "metricNames" section of the report.
	Metrics []*Metric `json:"metrics,omitempty"`

	// PivotedActivityMetrics: Metrics which are compatible to be selected
	// as activity metrics to pivot on in the "activities" section of the
	// report.
	PivotedActivityMetrics []*Metric `json:"pivotedActivityMetrics,omitempty"`

	// ReachByFrequencyMetrics: Metrics which are compatible to be selected
	// in the "reachByFrequencyMetricNames" section of the report.
	ReachByFrequencyMetrics []*Metric `json:"reachByFrequencyMetrics,omitempty"`
}

type Recipient struct {
	// DeliveryType: The delivery type for the recipient, one of:
	// -
	// "ATTACHMENT"
	// - "LINK"
	DeliveryType string `json:"deliveryType,omitempty"`

	// Email: The email address of the recipient.
	Email string `json:"email,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#recipient.
	Kind string `json:"kind,omitempty"`
}

type Report struct {
	// AccountId: The account ID to which this report belongs.
	AccountId int64 `json:"accountId,omitempty,string"`

	// ActiveGrpCriteria: The report criteria for a report of type
	// "ACTIVE_GRP".
	ActiveGrpCriteria *ReportActiveGrpCriteria `json:"activeGrpCriteria,omitempty"`

	// Criteria: The report criteria for a report of type "STANDARD".
	Criteria *ReportCriteria `json:"criteria,omitempty"`

	// CrossDimensionReachCriteria: The report criteria for a report of type
	// "CROSS_DIMENSION_REACH".
	CrossDimensionReachCriteria *ReportCrossDimensionReachCriteria `json:"crossDimensionReachCriteria,omitempty"`

	// Delivery: The report's email delivery settings.
	Delivery *ReportDelivery `json:"delivery,omitempty"`

	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// FileName: The file name used when generating report files for this
	// report.
	FileName string `json:"fileName,omitempty"`

	// FloodlightCriteria: The report criteria for a report of type
	// "FLOODLIGHT".
	FloodlightCriteria *ReportFloodlightCriteria `json:"floodlightCriteria,omitempty"`

	// Format: The output format of the report, one of:
	// - "CSV"
	// - "EXCEL"
	//  If not specified, default format is "CSV". Note that the actual
	// format in the completed report file might differ if for instance the
	// report's size exceeds the format's capabilities. "CSV" will then be
	// the fallback format.
	Format string `json:"format,omitempty"`

	// Id: The unique ID identifying this report resource.
	Id int64 `json:"id,omitempty,string"`

	// Kind: The kind of resource this is, in this case dfareporting#report.
	Kind string `json:"kind,omitempty"`

	// LastModifiedTime: The timestamp (in milliseconds since epoch) of when
	// this report was last modified.
	LastModifiedTime uint64 `json:"lastModifiedTime,omitempty,string"`

	// Name: The name of the report.
	Name string `json:"name,omitempty"`

	// OwnerProfileId: The user profile id of the owner of this report.
	OwnerProfileId int64 `json:"ownerProfileId,omitempty,string"`

	// PathToConversionCriteria: The report criteria for a report of type
	// "PATH_TO_CONVERSION".
	PathToConversionCriteria *ReportPathToConversionCriteria `json:"pathToConversionCriteria,omitempty"`

	// ReachCriteria: The report criteria for a report of type "REACH".
	ReachCriteria *ReportReachCriteria `json:"reachCriteria,omitempty"`

	// Schedule: The report's schedule. Can only be set if the report's
	// 'dateRange' is a relative date range and the relative date range is
	// not "TODAY".
	Schedule *ReportSchedule `json:"schedule,omitempty"`

	// SubAccountId: The subbaccount ID to which this report belongs if
	// applicable.
	SubAccountId int64 `json:"subAccountId,omitempty,string"`

	// Type: The type of the report, one of:
	// - STANDARD
	// - REACH
	// -
	// ACTIVE_GRP
	// - PATH_TO_CONVERSION
	// - FLOODLIGHT
	// -
	// CROSS_DIMENSION_REACH
	Type string `json:"type,omitempty"`
}

type ReportActiveGrpCriteria struct {
	// DateRange: The date range this report should be run for.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// DimensionFilters: The list of filters on which dimensions are
	// filtered.
	// Filters for different dimensions are ANDed, filters for the
	// same dimension are grouped together and ORed.
	// A valid active GRP
	// report needs to have exactly one DimensionValue for the United States
	// in addition to any advertiser or campaign dimension values.
	DimensionFilters []*DimensionValue `json:"dimensionFilters,omitempty"`

	// Dimensions: The list of dimensions the report should include.
	Dimensions []*SortedDimension `json:"dimensions,omitempty"`

	// MetricNames: The list of names of metrics the report should include.
	MetricNames []string `json:"metricNames,omitempty"`
}

type ReportCriteria struct {
	// Activities: Activity group.
	Activities *Activities `json:"activities,omitempty"`

	// CustomRichMediaEvents: Custom Rich Media Events group.
	CustomRichMediaEvents *CustomRichMediaEvents `json:"customRichMediaEvents,omitempty"`

	// DateRange: The date range for which this report should be run.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// DimensionFilters: The list of filters on which dimensions are
	// filtered.
	// Filters for different dimensions are ANDed, filters for the
	// same dimension are grouped together and ORed.
	DimensionFilters []*DimensionValue `json:"dimensionFilters,omitempty"`

	// Dimensions: The list of standard dimensions the report should
	// include.
	Dimensions []*SortedDimension `json:"dimensions,omitempty"`

	// MetricNames: The list of names of metrics the report should include.
	MetricNames []string `json:"metricNames,omitempty"`
}

type ReportCrossDimensionReachCriteria struct {
	// Breakdown: The list of dimensions the report should include.
	Breakdown []*SortedDimension `json:"breakdown,omitempty"`

	// DateRange: The date range this report should be run for.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// Dimension: The dimension option, one of:
	// - "ADVERTISER"
	// -
	// "CAMPAIGN"
	// - "SITE_BY_ADVERTISER"
	// - "SITE_BY_CAMPAIGN"
	Dimension string `json:"dimension,omitempty"`

	// DimensionFilters: The list of filters on which dimensions are
	// filtered.
	DimensionFilters []*DimensionValue `json:"dimensionFilters,omitempty"`

	// MetricNames: The list of names of metrics the report should include.
	MetricNames []string `json:"metricNames,omitempty"`

	// OverlapMetricNames: The list of names of overlap metrics the report
	// should include.
	OverlapMetricNames []string `json:"overlapMetricNames,omitempty"`

	// Pivoted: Whether the report is pivoted or not. Defaults to true.
	Pivoted bool `json:"pivoted,omitempty"`
}

type ReportDelivery struct {
	// EmailOwner: Whether the report should be emailed to the report owner.
	EmailOwner bool `json:"emailOwner,omitempty"`

	// EmailOwnerDeliveryType: The type of delivery for the owner to
	// receive, if enabled. One of:
	// - "ATTACHMENT"
	// - "LINK"
	EmailOwnerDeliveryType string `json:"emailOwnerDeliveryType,omitempty"`

	// Message: The message to be sent with each email.
	Message string `json:"message,omitempty"`

	// Recipients: The list of recipients to which to email the report.
	Recipients []*Recipient `json:"recipients,omitempty"`
}

type ReportFloodlightCriteria struct {
	// CustomRichMediaEvents: The list of custom rich media events to
	// include.
	CustomRichMediaEvents []*DimensionValue `json:"customRichMediaEvents,omitempty"`

	// DateRange: The date range this report should be run for.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// DimensionFilters: The list of filters on which dimensions are
	// filtered.
	// Filters for different dimensions are ANDed, filters for the
	// same dimension are grouped together and ORed.
	DimensionFilters []*DimensionValue `json:"dimensionFilters,omitempty"`

	// Dimensions: The list of dimensions the report should include.
	Dimensions []*SortedDimension `json:"dimensions,omitempty"`

	// FloodlightConfigId: The floodlight ID for which to show data in this
	// report. All advertisers associated with that ID will automatically be
	// added. The dimension of the value needs to be
	// 'dfa:floodlightConfigId'.
	FloodlightConfigId *DimensionValue `json:"floodlightConfigId,omitempty"`

	// MetricNames: The list of names of metrics the report should include.
	MetricNames []string `json:"metricNames,omitempty"`

	// ReportProperties: The properties of the report.
	ReportProperties *ReportFloodlightCriteriaReportProperties `json:"reportProperties,omitempty"`
}

type ReportFloodlightCriteriaReportProperties struct {
	// IncludeAttributedIPConversions: Include conversions that have no
	// cookie, but do have an exposure path.
	IncludeAttributedIPConversions bool `json:"includeAttributedIPConversions,omitempty"`

	// IncludeUnattributedCookieConversions: Include conversions of users
	// with a DoubleClick cookie but without an exposure. That means the
	// user did not click or see an ad from the advertiser within the
	// Floodlight group, or that the interaction happened outside the
	// lookback window.
	IncludeUnattributedCookieConversions bool `json:"includeUnattributedCookieConversions,omitempty"`

	// IncludeUnattributedIPConversions: Include conversions that have no
	// associated cookies and no exposures. It’s therefore impossible to
	// know how the user was exposed to your ads during the lookback window
	// prior to a conversion.
	IncludeUnattributedIPConversions bool `json:"includeUnattributedIPConversions,omitempty"`
}

type ReportPathToConversionCriteria struct {
	// ActivityFilters: The list of 'dfa:activity' values to filter on.
	ActivityFilters []*DimensionValue `json:"activityFilters,omitempty"`

	// ConversionDimensions: The list of conversion dimensions the report
	// should include.
	ConversionDimensions []*SortedDimension `json:"conversionDimensions,omitempty"`

	// CustomFloodlightVariables: The list of custom floodlight variables
	// the report should include.
	CustomFloodlightVariables []*SortedDimension `json:"customFloodlightVariables,omitempty"`

	// CustomRichMediaEvents: The list of custom rich media events to
	// include.
	CustomRichMediaEvents []*DimensionValue `json:"customRichMediaEvents,omitempty"`

	// DateRange: The date range this report should be run for.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// FloodlightConfigId: The floodlight ID for which to show data in this
	// report. All advertisers associated with that ID will automatically be
	// added. The dimension of the value needs to be
	// 'dfa:floodlightConfigId'.
	FloodlightConfigId *DimensionValue `json:"floodlightConfigId,omitempty"`

	// MetricNames: The list of names of metrics the report should include.
	MetricNames []string `json:"metricNames,omitempty"`

	// PerInteractionDimensions: The list of per interaction dimensions the
	// report should include.
	PerInteractionDimensions []*SortedDimension `json:"perInteractionDimensions,omitempty"`

	// ReportProperties: The properties of the report.
	ReportProperties *ReportPathToConversionCriteriaReportProperties `json:"reportProperties,omitempty"`
}

type ReportPathToConversionCriteriaReportProperties struct {
	// ClicksLookbackWindow: DFA checks to see if a click interaction
	// occurred within the specified period of time before a conversion. By
	// default the value is pulled from Floodlight or you can manually enter
	// a custom value. Valid values: 1-90.
	ClicksLookbackWindow int64 `json:"clicksLookbackWindow,omitempty"`

	// ImpressionsLookbackWindow: DFA checks to see if an impression
	// interaction occurred within the specified period of time before a
	// conversion. By default the value is pulled from Floodlight or you can
	// manually enter a custom value. Valid values: 1-90.
	ImpressionsLookbackWindow int64 `json:"impressionsLookbackWindow,omitempty"`

	// IncludeAttributedIPConversions: Deprecated: has no effect.
	IncludeAttributedIPConversions bool `json:"includeAttributedIPConversions,omitempty"`

	// IncludeUnattributedCookieConversions: Include conversions of users
	// with a DoubleClick cookie but without an exposure. That means the
	// user did not click or see an ad from the advertiser within the
	// Floodlight group, or that the interaction happened outside the
	// lookback window.
	IncludeUnattributedCookieConversions bool `json:"includeUnattributedCookieConversions,omitempty"`

	// IncludeUnattributedIPConversions: Include conversions that have no
	// associated cookies and no exposures. It’s therefore impossible to
	// know how the user was exposed to your ads during the lookback window
	// prior to a conversion.
	IncludeUnattributedIPConversions bool `json:"includeUnattributedIPConversions,omitempty"`

	// MaximumClickInteractions: The maximum number of click interactions to
	// include in the report. Advertisers currently paying for E2C reports
	// get up to 200 (100 clicks, 100 impressions). If another advertiser in
	// your network is paying for E2C, you can have up to 5 total exposures
	// per report.
	MaximumClickInteractions int64 `json:"maximumClickInteractions,omitempty"`

	// MaximumImpressionInteractions: The maximum number of click
	// interactions to include in the report. Advertisers currently paying
	// for E2C reports get up to 200 (100 clicks, 100 impressions). If
	// another advertiser in your network is paying for E2C, you can have up
	// to 5 total exposures per report.
	MaximumImpressionInteractions int64 `json:"maximumImpressionInteractions,omitempty"`

	// MaximumInteractionGap: The maximum amount of time that can take place
	// between interactions (clicks or impressions) by the same user. Valid
	// values: 1-90.
	MaximumInteractionGap int64 `json:"maximumInteractionGap,omitempty"`

	// PivotOnInteractionPath: Enable pivoting on interaction path.
	PivotOnInteractionPath bool `json:"pivotOnInteractionPath,omitempty"`
}

type ReportReachCriteria struct {
	// Activities: Activity group.
	Activities *Activities `json:"activities,omitempty"`

	// CustomRichMediaEvents: Custom Rich Media Events group.
	CustomRichMediaEvents *CustomRichMediaEvents `json:"customRichMediaEvents,omitempty"`

	// DateRange: The date range this report should be run for.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// DimensionFilters: The list of filters on which dimensions are
	// filtered.
	// Filters for different dimensions are ANDed, filters for the
	// same dimension are grouped together and ORed.
	DimensionFilters []*DimensionValue `json:"dimensionFilters,omitempty"`

	// Dimensions: The list of dimensions the report should include.
	Dimensions []*SortedDimension `json:"dimensions,omitempty"`

	// MetricNames: The list of names of metrics the report should include.
	MetricNames []string `json:"metricNames,omitempty"`

	// ReachByFrequencyMetricNames: The list of names of  Reach By Frequency
	// metrics the report should include.
	ReachByFrequencyMetricNames []string `json:"reachByFrequencyMetricNames,omitempty"`
}

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

	// Repeats: The interval for which the report is repeated, one of:
	// -
	// "DAILY", also requires field "every" to be set.
	// - "WEEKLY", also
	// requires fields "every" and "repeatsOnWeekDays" to be set.
	// -
	// "TWICE_A_MONTH"
	// - "MONTHLY", also requires fields "every" and
	// "runsOnDayOfMonth" to be set.
	// - "QUARTERLY"
	// - "YEARLY"
	Repeats string `json:"repeats,omitempty"`

	// RepeatsOnWeekDays: List of week days "WEEKLY" on which scheduled
	// reports should run.
	RepeatsOnWeekDays []string `json:"repeatsOnWeekDays,omitempty"`

	// RunsOnDayOfMonth: Enum to define for "MONTHLY" scheduled reports
	// whether reports should be repeated on the same day of the month as
	// "startDate" or the same day of the week of the month. Possible values
	// are:
	// - DAY_OF_MONTH
	// - WEEK_OF_MONTH
	// Example: If 'startDate' is
	// Monday, April 2nd 2012 (2012-04-02), "DAY_OF_MONTH" would run
	// subsequent reports on the 2nd of every Month, and "WEEK_OF_MONTH"
	// would run subsequent reports on the first Monday of the month.
	RunsOnDayOfMonth string `json:"runsOnDayOfMonth,omitempty"`

	// StartDate: Start date of date range for which scheduled reports
	// should be run.
	StartDate string `json:"startDate,omitempty"`
}

type ReportCompatibleFields struct {
	// DimensionFilters: Dimensions which are compatible to be selected in
	// the "dimensionFilters" section of the report.
	DimensionFilters []*Dimension `json:"dimensionFilters,omitempty"`

	// Dimensions: Dimensions which are compatible to be selected in the
	// "dimensions" section of the report.
	Dimensions []*Dimension `json:"dimensions,omitempty"`

	// Kind: The kind of resource this is, in this case
	// dfareporting#reportCompatibleFields.
	Kind string `json:"kind,omitempty"`

	// Metrics: Metrics which are compatible to be selected in the
	// "metricNames" section of the report.
	Metrics []*Metric `json:"metrics,omitempty"`

	// PivotedActivityMetrics: Metrics which are compatible to be selected
	// as activity metrics to pivot on in the "activities" section of the
	// report.
	PivotedActivityMetrics []*Metric `json:"pivotedActivityMetrics,omitempty"`
}

type ReportList struct {
	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The reports returned in this response.
	Items []*Report `json:"items,omitempty"`

	// Kind: The kind of list this is, in this case dfareporting#reportList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through reports. To
	// retrieve the next page of results, set the next request's "pageToken"
	// to the value of this field. The page token is only valid for a
	// limited amount of time and should not be persisted.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type SortedDimension struct {
	// Kind: The kind of resource this is, in this case
	// dfareporting#sortedDimension.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the dimension.
	Name string `json:"name,omitempty"`

	// SortOrder: An optional sort order for the dimension column, one of:
	//
	// - "ASCENDING"
	// - "DESCENDING"
	SortOrder string `json:"sortOrder,omitempty"`
}

type UserProfile struct {
	// AccountId: The account ID to which this profile belongs.
	AccountId int64 `json:"accountId,omitempty,string"`

	// AccountName: The account name this profile belongs to.
	AccountName string `json:"accountName,omitempty"`

	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Kind: The kind of resource this is, in this case
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
}

type UserProfileList struct {
	// Etag: The eTag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The user profiles returned in this response.
	Items []*UserProfile `json:"items,omitempty"`

	// Kind: The kind of list this is, in this case
	// dfareporting#userProfileList.
	Kind string `json:"kind,omitempty"`
}

// method id "dfareporting.dimensionValues.query":

type DimensionValuesQueryCall struct {
	s                     *Service
	profileId             int64
	dimensionvaluerequest *DimensionValueRequest
	opt_                  map[string]interface{}
}

// Query: Retrieves list of report dimension values for a list of
// filters.
func (r *DimensionValuesService) Query(profileId int64, dimensionvaluerequest *DimensionValueRequest) *DimensionValuesQueryCall {
	c := &DimensionValuesQueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.dimensionvaluerequest = dimensionvaluerequest
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *DimensionValuesQueryCall) MaxResults(maxResults int64) *DimensionValuesQueryCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *DimensionValuesQueryCall) PageToken(pageToken string) *DimensionValuesQueryCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *DimensionValuesQueryCall) Do() (*DimensionValueList, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dimensionvaluerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/dimensionvalues/query")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
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
	ret := new(DimensionValueList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA user profile ID.",
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

// method id "dfareporting.files.get":

type FilesGetCall struct {
	s        *Service
	reportId int64
	fileId   int64
	opt_     map[string]interface{}
}

// Get: Retrieves a report file by its report ID and file ID.
func (r *FilesService) Get(reportId int64, fileId int64) *FilesGetCall {
	c := &FilesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.reportId = reportId
	c.fileId = fileId
	return c
}

func (c *FilesGetCall) Do() (*File, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/{reportId}/files/{fileId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{fileId}", strconv.FormatInt(c.fileId, 10), 1)
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
	ret := new(File)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a report file by its report ID and file ID.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.files.get",
	//   "parameterOrder": [
	//     "reportId",
	//     "fileId"
	//   ],
	//   "parameters": {
	//     "fileId": {
	//       "description": "The ID of the report file.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "reports/{reportId}/files/{fileId}",
	//   "response": {
	//     "$ref": "File"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "dfareporting.files.list":

type FilesListCall struct {
	s         *Service
	profileId int64
	opt_      map[string]interface{}
}

// List: Lists files for a user profile.
func (r *FilesService) List(profileId int64) *FilesListCall {
	c := &FilesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *FilesListCall) MaxResults(maxResults int64) *FilesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *FilesListCall) PageToken(pageToken string) *FilesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Scope sets the optional parameter "scope": The scope that defines
// which results are returned, default is 'MINE'.
func (c *FilesListCall) Scope(scope string) *FilesListCall {
	c.opt_["scope"] = scope
	return c
}

// SortField sets the optional parameter "sortField": The field by which
// to sort the list.
func (c *FilesListCall) SortField(sortField string) *FilesListCall {
	c.opt_["sortField"] = sortField
	return c
}

// SortOrder sets the optional parameter "sortOrder": Order of sorted
// results, default is 'DESCENDING'.
func (c *FilesListCall) SortOrder(sortOrder string) *FilesListCall {
	c.opt_["sortOrder"] = sortOrder
	return c
}

func (c *FilesListCall) Do() (*FileList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["scope"]; ok {
		params.Set("scope", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortField"]; ok {
		params.Set("sortField", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortOrder"]; ok {
		params.Set("sortOrder", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/files")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
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
	ret := new(FileList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "scope": {
	//       "default": "MINE",
	//       "description": "The scope that defines which results are returned, default is 'MINE'.",
	//       "enum": [
	//         "ALL",
	//         "MINE",
	//         "SHARED_WITH_ME"
	//       ],
	//       "enumDescriptions": [
	//         "All files in account.",
	//         "My files.",
	//         "Files shared with me."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sortField": {
	//       "default": "LAST_MODIFIED_TIME",
	//       "description": "The field by which to sort the list.",
	//       "enum": [
	//         "ID",
	//         "LAST_MODIFIED_TIME"
	//       ],
	//       "enumDescriptions": [
	//         "Sort by file ID.",
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
	s         *Service
	profileId int64
	reportId  int64
	opt_      map[string]interface{}
}

// Delete: Deletes a report by its ID.
func (r *ReportsService) Delete(profileId int64, reportId int64) *ReportsDeleteCall {
	c := &ReportsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

func (c *ReportsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
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
	//   "description": "Deletes a report by its ID.",
	//   "httpMethod": "DELETE",
	//   "id": "dfareporting.reports.delete",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report.",
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
	s         *Service
	profileId int64
	reportId  int64
	opt_      map[string]interface{}
}

// Get: Retrieves a report by its ID.
func (r *ReportsService) Get(profileId int64, reportId int64) *ReportsGetCall {
	c := &ReportsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

func (c *ReportsGetCall) Do() (*Report, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
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
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a report by its ID.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.reports.get",
	//   "parameterOrder": [
	//     "profileId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report.",
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
	s         *Service
	profileId int64
	report    *Report
	opt_      map[string]interface{}
}

// Insert: Creates a report.
func (r *ReportsService) Insert(profileId int64, report *Report) *ReportsInsertCall {
	c := &ReportsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.report = report
	return c
}

func (c *ReportsInsertCall) Do() (*Report, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.report)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
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
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA user profile ID.",
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
	s         *Service
	profileId int64
	opt_      map[string]interface{}
}

// List: Retrieves list of reports.
func (r *ReportsService) List(profileId int64) *ReportsListCall {
	c := &ReportsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *ReportsListCall) MaxResults(maxResults int64) *ReportsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *ReportsListCall) PageToken(pageToken string) *ReportsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Scope sets the optional parameter "scope": The scope that defines
// which results are returned, default is 'MINE'.
func (c *ReportsListCall) Scope(scope string) *ReportsListCall {
	c.opt_["scope"] = scope
	return c
}

// SortField sets the optional parameter "sortField": The field by which
// to sort the list.
func (c *ReportsListCall) SortField(sortField string) *ReportsListCall {
	c.opt_["sortField"] = sortField
	return c
}

// SortOrder sets the optional parameter "sortOrder": Order of sorted
// results, default is 'DESCENDING'.
func (c *ReportsListCall) SortOrder(sortOrder string) *ReportsListCall {
	c.opt_["sortOrder"] = sortOrder
	return c
}

func (c *ReportsListCall) Do() (*ReportList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["scope"]; ok {
		params.Set("scope", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortField"]; ok {
		params.Set("sortField", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortOrder"]; ok {
		params.Set("sortOrder", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
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
	ret := new(ReportList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA user profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "scope": {
	//       "default": "MINE",
	//       "description": "The scope that defines which results are returned, default is 'MINE'.",
	//       "enum": [
	//         "ALL",
	//         "MINE"
	//       ],
	//       "enumDescriptions": [
	//         "All reports in account.",
	//         "My reports."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sortField": {
	//       "default": "LAST_MODIFIED_TIME",
	//       "description": "The field by which to sort the list.",
	//       "enum": [
	//         "ID",
	//         "LAST_MODIFIED_TIME",
	//         "NAME"
	//       ],
	//       "enumDescriptions": [
	//         "Sort by report ID.",
	//         "Sort by 'lastModifiedTime' field.",
	//         "Sort by name of reports."
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
	s         *Service
	profileId int64
	reportId  int64
	report    *Report
	opt_      map[string]interface{}
}

// Patch: Updates a report. This method supports patch semantics.
func (r *ReportsService) Patch(profileId int64, reportId int64, report *Report) *ReportsPatchCall {
	c := &ReportsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.reportId = reportId
	c.report = report
	return c
}

func (c *ReportsPatchCall) Do() (*Report, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.report)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
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
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA user profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report.",
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
	s         *Service
	profileId int64
	reportId  int64
	opt_      map[string]interface{}
}

// Run: Runs a report.
func (r *ReportsService) Run(profileId int64, reportId int64) *ReportsRunCall {
	c := &ReportsRunCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

// Synchronous sets the optional parameter "synchronous": If set and
// true, tries to run the report synchronously.
func (c *ReportsRunCall) Synchronous(synchronous bool) *ReportsRunCall {
	c.opt_["synchronous"] = synchronous
	return c
}

func (c *ReportsRunCall) Do() (*File, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["synchronous"]; ok {
		params.Set("synchronous", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}/run")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
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
	ret := new(File)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report.",
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
	s         *Service
	profileId int64
	reportId  int64
	report    *Report
	opt_      map[string]interface{}
}

// Update: Updates a report.
func (r *ReportsService) Update(profileId int64, reportId int64, report *Report) *ReportsUpdateCall {
	c := &ReportsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.reportId = reportId
	c.report = report
	return c
}

func (c *ReportsUpdateCall) Do() (*Report, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.report)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
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
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA user profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report.",
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

// method id "dfareporting.reports.compatibleFields.query":

type ReportsCompatibleFieldsQueryCall struct {
	s         *Service
	profileId int64
	report    *Report
	opt_      map[string]interface{}
}

// Query: Returns the fields that are compatible to be selected in the
// respective sections of a report criteria, given the fields already
// selected in the input report and user permissions.
func (r *ReportsCompatibleFieldsService) Query(profileId int64, report *Report) *ReportsCompatibleFieldsQueryCall {
	c := &ReportsCompatibleFieldsQueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.report = report
	return c
}

func (c *ReportsCompatibleFieldsQueryCall) Do() (*CompatibleFields, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.report)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/compatiblefields/query")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
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
	ret := new(CompatibleFields)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the fields that are compatible to be selected in the respective sections of a report criteria, given the fields already selected in the input report and user permissions.",
	//   "httpMethod": "POST",
	//   "id": "dfareporting.reports.compatibleFields.query",
	//   "parameterOrder": [
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The DFA user profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "userprofiles/{profileId}/reports/compatiblefields/query",
	//   "request": {
	//     "$ref": "Report"
	//   },
	//   "response": {
	//     "$ref": "CompatibleFields"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/dfareporting"
	//   ]
	// }

}

// method id "dfareporting.reports.files.get":

type ReportsFilesGetCall struct {
	s         *Service
	profileId int64
	reportId  int64
	fileId    int64
	opt_      map[string]interface{}
}

// Get: Retrieves a report file.
func (r *ReportsFilesService) Get(profileId int64, reportId int64, fileId int64) *ReportsFilesGetCall {
	c := &ReportsFilesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.reportId = reportId
	c.fileId = fileId
	return c
}

func (c *ReportsFilesGetCall) Do() (*File, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}/files/{fileId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{fileId}", strconv.FormatInt(c.fileId, 10), 1)
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
	ret := new(File)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The ID of the report file.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "The DFA profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report.",
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
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "dfareporting.reports.files.list":

type ReportsFilesListCall struct {
	s         *Service
	profileId int64
	reportId  int64
	opt_      map[string]interface{}
}

// List: Lists files for a report.
func (r *ReportsFilesService) List(profileId int64, reportId int64) *ReportsFilesListCall {
	c := &ReportsFilesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	c.reportId = reportId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *ReportsFilesListCall) MaxResults(maxResults int64) *ReportsFilesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous result page.
func (c *ReportsFilesListCall) PageToken(pageToken string) *ReportsFilesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// SortField sets the optional parameter "sortField": The field by which
// to sort the list.
func (c *ReportsFilesListCall) SortField(sortField string) *ReportsFilesListCall {
	c.opt_["sortField"] = sortField
	return c
}

// SortOrder sets the optional parameter "sortOrder": Order of sorted
// results, default is 'DESCENDING'.
func (c *ReportsFilesListCall) SortOrder(sortOrder string) *ReportsFilesListCall {
	c.opt_["sortOrder"] = sortOrder
	return c
}

func (c *ReportsFilesListCall) Do() (*FileList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortField"]; ok {
		params.Set("sortField", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortOrder"]; ok {
		params.Set("sortOrder", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}/reports/{reportId}/files")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", strconv.FormatInt(c.reportId, 10), 1)
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
	ret := new(FileList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "The DFA profile ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the parent report.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sortField": {
	//       "default": "LAST_MODIFIED_TIME",
	//       "description": "The field by which to sort the list.",
	//       "enum": [
	//         "ID",
	//         "LAST_MODIFIED_TIME"
	//       ],
	//       "enumDescriptions": [
	//         "Sort by file ID.",
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
	s         *Service
	profileId int64
	opt_      map[string]interface{}
}

// Get: Gets one user profile by ID.
func (r *UserProfilesService) Get(profileId int64) *UserProfilesGetCall {
	c := &UserProfilesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.profileId = profileId
	return c
}

func (c *UserProfilesGetCall) Do() (*UserProfile, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles/{profileId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", strconv.FormatInt(c.profileId, 10), 1)
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
	ret := new(UserProfile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one user profile by ID.",
	//   "httpMethod": "GET",
	//   "id": "dfareporting.userProfiles.get",
	//   "parameterOrder": [
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "profileId": {
	//       "description": "The user profile ID.",
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
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves list of user profiles for a user.
func (r *UserProfilesService) List() *UserProfilesListCall {
	c := &UserProfilesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *UserProfilesListCall) Do() (*UserProfileList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "userprofiles")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
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
	ret := new(UserProfileList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
