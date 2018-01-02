// Package analyticsreporting provides access to the Google Analytics Reporting API.
//
// See https://developers.google.com/analytics/devguides/reporting/core/v4/
//
// Usage example:
//
//   import "google.golang.org/api/analyticsreporting/v4"
//   ...
//   analyticsreportingService, err := analyticsreporting.New(oauthHttpClient)
package analyticsreporting // import "google.golang.org/api/analyticsreporting/v4"

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

const apiId = "analyticsreporting:v4"
const apiName = "analyticsreporting"
const apiVersion = "v4"
const basePath = "https://analyticsreporting.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Google Analytics data
	AnalyticsScope = "https://www.googleapis.com/auth/analytics"

	// View your Google Analytics data
	AnalyticsReadonlyScope = "https://www.googleapis.com/auth/analytics.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Reports = NewReportsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Reports *ReportsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	return rs
}

type ReportsService struct {
	s *Service
}

// Cohort: Defines a cohort. A cohort is a group of users who share a
// common
// characteristic. For example, all users with the same acquisition
// date
// belong to the same cohort.
type Cohort struct {
	// DateRange: This is used for `FIRST_VISIT_DATE` cohort, the cohort
	// selects users
	// whose first visit date is between start date and end date defined in
	// the
	// DateRange. The date ranges should be aligned for cohort requests. If
	// the
	// request contains `ga:cohortNthDay` it should be exactly one day
	// long,
	// if `ga:cohortNthWeek` it should be aligned to the week boundary
	// (starting
	// at Sunday and ending Saturday), and for `ga:cohortNthMonth` the date
	// range
	// should be aligned to the month (starting at the first and ending on
	// the
	// last day of the month).
	// For LTV requests there are no such restrictions.
	// You do not need to supply a date range for
	// the
	// `reportsRequest.dateRanges` field.
	DateRange *DateRange `json:"dateRange,omitempty"`

	// Name: A unique name for the cohort. If not defined name will be
	// auto-generated
	// with values cohort_[1234...].
	Name string `json:"name,omitempty"`

	// Type: Type of the cohort. The only supported type as of now
	// is
	// `FIRST_VISIT_DATE`. If this field is unspecified the cohort is
	// treated
	// as `FIRST_VISIT_DATE` type cohort.
	//
	// Possible values:
	//   "UNSPECIFIED_COHORT_TYPE" - If unspecified it's treated as
	// `FIRST_VISIT_DATE`.
	//   "FIRST_VISIT_DATE" - Cohorts that are selected based on first visit
	// date.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DateRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DateRange") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Cohort) MarshalJSON() ([]byte, error) {
	type noMethod Cohort
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CohortGroup: Defines a cohort group.
// For example:
//
//     "cohortGroup": {
//       "cohorts": [{
//         "name": "cohort 1",
//         "type": "FIRST_VISIT_DATE",
//         "dateRange": { "startDate": "2015-08-01", "endDate":
// "2015-08-01" }
//       },{
//         "name": "cohort 2"
//          "type": "FIRST_VISIT_DATE"
//          "dateRange": { "startDate": "2015-07-01", "endDate":
// "2015-07-01" }
//       }]
//     }
type CohortGroup struct {
	// Cohorts: The definition for the cohort.
	Cohorts []*Cohort `json:"cohorts,omitempty"`

	// LifetimeValue: Enable Life Time Value (LTV).  LTV measures lifetime
	// value for users
	// acquired through different channels.
	// Please see:
	// [Cohort
	// Analysis](https://support.google.com/analytics/answer/6074676)
	// and
	// [Lifetime
	// Value](https://support.google.com/analytics/answer/6182550)
	// If the value of lifetimeValue is false:
	//
	// - The metric values are similar to the values in the web interface
	// cohort
	//   report.
	// - The cohort definition date ranges must be aligned to the calendar
	// week
	//   and month. i.e. while requesting `ga:cohortNthWeek` the `startDate`
	// in
	//   the cohort definition should be a Sunday and the `endDate` should
	// be the
	//   following Saturday, and for `ga:cohortNthMonth`, the `startDate`
	//   should be the 1st of the month and `endDate` should be the last
	// day
	//   of the month.
	//
	// When the lifetimeValue is true:
	//
	// - The metric values will correspond to the values in the web
	// interface
	//   LifeTime value report.
	// - The Lifetime Value report shows you how user value (Revenue) and
	//   engagement (Appviews, Goal Completions, Sessions, and Session
	// Duration)
	//   grow during the 90 days after a user is acquired.
	// - The metrics are calculated as a cumulative average per user per the
	// time
	//   increment.
	// - The cohort definition date ranges need not be aligned to the
	// calendar
	//   week and month boundaries.
	// - The `viewId` must be an
	//   [app view
	// ID](https://support.google.com/analytics/answer/2649553#WebVersusAppVi
	// ews)
	LifetimeValue bool `json:"lifetimeValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cohorts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cohorts") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CohortGroup) MarshalJSON() ([]byte, error) {
	type noMethod CohortGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ColumnHeader: Column headers.
type ColumnHeader struct {
	// Dimensions: The dimension names in the response.
	Dimensions []string `json:"dimensions,omitempty"`

	// MetricHeader: Metric headers for the metrics in the response.
	MetricHeader *MetricHeader `json:"metricHeader,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Dimensions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Dimensions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ColumnHeader) MarshalJSON() ([]byte, error) {
	type noMethod ColumnHeader
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DateRange: A contiguous set of days: startDate, startDate + 1 day,
// ..., endDate.
// The start and end dates are specified
// in
// [ISO8601](https://en.wikipedia.org/wiki/ISO_8601) date format
// `YYYY-MM-DD`.
type DateRange struct {
	// EndDate: The end date for the query in the format `YYYY-MM-DD`.
	EndDate string `json:"endDate,omitempty"`

	// StartDate: The start date for the query in the format `YYYY-MM-DD`.
	StartDate string `json:"startDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndDate") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DateRange) MarshalJSON() ([]byte, error) {
	type noMethod DateRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DateRangeValues: Used to return a list of metrics for a single
// DateRange / dimension
// combination
type DateRangeValues struct {
	// PivotValueRegions: The values of each pivot region.
	PivotValueRegions []*PivotValueRegion `json:"pivotValueRegions,omitempty"`

	// Values: Each value corresponds to each Metric in the request.
	Values []string `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PivotValueRegions")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PivotValueRegions") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DateRangeValues) MarshalJSON() ([]byte, error) {
	type noMethod DateRangeValues
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Dimension:
// [Dimensions](https://support.google.com/analytics/answer/1033861)
// are attributes of your data. For example, the dimension
// `ga:city`
// indicates the city, for example, "Paris" or "New York", from which
// a session originates.
type Dimension struct {
	// HistogramBuckets: If non-empty, we place dimension values into
	// buckets after string to
	// int64. Dimension values that are not the string representation of
	// an
	// integral value will be converted to zero.  The bucket values have to
	// be in
	// increasing order.  Each bucket is closed on the lower end, and open
	// on the
	// upper end. The "first" bucket includes all values less than the
	// first
	// boundary, the "last" bucket includes all values up to infinity.
	// Dimension
	// values that fall in a bucket get transformed to a new dimension
	// value. For
	// example, if one gives a list of "0, 1, 3, 4, 7", then we return
	// the
	// following buckets:
	//
	// - bucket #1: values < 0, dimension value "<0"
	// - bucket #2: values in [0,1), dimension value "0"
	// - bucket #3: values in [1,3), dimension value "1-2"
	// - bucket #4: values in [3,4), dimension value "3"
	// - bucket #5: values in [4,7), dimension value "4-6"
	// - bucket #6: values >= 7, dimension value "7+"
	//
	// NOTE: If you are applying histogram mutation on any dimension, and
	// using
	// that dimension in sort, you will want to use the sort
	// type
	// `HISTOGRAM_BUCKET` for that purpose. Without that the dimension
	// values
	// will be sorted according to dictionary
	// (lexicographic) order. For example the ascending dictionary order
	// is:
	//
	//    "<50", "1001+", "121-1000", "50-120"
	//
	// And the ascending `HISTOGRAM_BUCKET` order is:
	//
	//    "<50", "50-120", "121-1000", "1001+"
	//
	// The client has to explicitly request "orderType":
	// "HISTOGRAM_BUCKET"
	// for a histogram-mutated dimension.
	HistogramBuckets googleapi.Int64s `json:"histogramBuckets,omitempty"`

	// Name: Name of the dimension to fetch, for example `ga:browser`.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HistogramBuckets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HistogramBuckets") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Dimension) MarshalJSON() ([]byte, error) {
	type noMethod Dimension
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DimensionFilter: Dimension filter specifies the filtering options on
// a dimension.
type DimensionFilter struct {
	// CaseSensitive: Should the match be case sensitive? Default is false.
	CaseSensitive bool `json:"caseSensitive,omitempty"`

	// DimensionName: The dimension to filter on. A DimensionFilter must
	// contain a dimension.
	DimensionName string `json:"dimensionName,omitempty"`

	// Expressions: Strings or regular expression to match against. Only the
	// first value of
	// the list is used for comparison unless the operator is `IN_LIST`.
	// If `IN_LIST` operator, then the entire list is used to filter
	// the
	// dimensions as explained in the description of the `IN_LIST` operator.
	Expressions []string `json:"expressions,omitempty"`

	// Not: Logical `NOT` operator. If this boolean is set to true, then the
	// matching
	// dimension values will be excluded in the report. The default is
	// false.
	Not bool `json:"not,omitempty"`

	// Operator: How to match the dimension to the expression. The default
	// is REGEXP.
	//
	// Possible values:
	//   "OPERATOR_UNSPECIFIED" - If the match type is unspecified, it is
	// treated as a `REGEXP`.
	//   "REGEXP" - The match expression is treated as a regular expression.
	// All match types
	// are not treated as regular expressions.
	//   "BEGINS_WITH" - Matches the value which begin with the match
	// expression provided.
	//   "ENDS_WITH" - Matches the values which end with the match
	// expression provided.
	//   "PARTIAL" - Substring match.
	//   "EXACT" - The value should match the match expression entirely.
	//   "NUMERIC_EQUAL" - Integer comparison filters.
	// case sensitivity is ignored for these and the expression
	// is assumed to be a string representing an integer.
	// Failure conditions:
	//
	// - If expression is not a valid int64, the client should expect
	//   an error.
	// - Input dimensions that are not valid int64 values will never match
	// the
	//   filter.
	//   "NUMERIC_GREATER_THAN" - Checks if the dimension is numerically
	// greater than the match
	// expression. Read the description for `NUMERIC_EQUALS` for
	// restrictions.
	//   "NUMERIC_LESS_THAN" - Checks if the dimension is numerically less
	// than the match expression.
	// Read the description for `NUMERIC_EQUALS` for restrictions.
	//   "IN_LIST" - This option is used to specify a dimension filter whose
	// expression can
	// take any value from a selected list of values. This helps
	// avoiding
	// evaluating multiple exact match dimension filters which are OR'ed
	// for
	// every single response row. For example:
	//
	//     expressions: ["A", "B", "C"]
	//
	// Any response row whose dimension has it is value as A, B or C,
	// matches
	// this DimensionFilter.
	Operator string `json:"operator,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CaseSensitive") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CaseSensitive") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DimensionFilter) MarshalJSON() ([]byte, error) {
	type noMethod DimensionFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DimensionFilterClause: A group of dimension filters. Set the operator
// value to specify how
// the filters are logically combined.
type DimensionFilterClause struct {
	// Filters: The repeated set of filters. They are logically combined
	// based on the
	// operator specified.
	Filters []*DimensionFilter `json:"filters,omitempty"`

	// Operator: The operator for combining multiple dimension filters. If
	// unspecified, it
	// is treated as an `OR`.
	//
	// Possible values:
	//   "OPERATOR_UNSPECIFIED" - Unspecified operator. It is treated as an
	// `OR`.
	//   "OR" - The logical `OR` operator.
	//   "AND" - The logical `AND` operator.
	Operator string `json:"operator,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Filters") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DimensionFilterClause) MarshalJSON() ([]byte, error) {
	type noMethod DimensionFilterClause
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DynamicSegment: Dynamic segment definition for defining the segment
// within the request.
// A segment can select users, sessions or both.
type DynamicSegment struct {
	// Name: The name of the dynamic segment.
	Name string `json:"name,omitempty"`

	// SessionSegment: Session Segment to select sessions to include in the
	// segment.
	SessionSegment *SegmentDefinition `json:"sessionSegment,omitempty"`

	// UserSegment: User Segment to select users to include in the segment.
	UserSegment *SegmentDefinition `json:"userSegment,omitempty"`

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

func (s *DynamicSegment) MarshalJSON() ([]byte, error) {
	type noMethod DynamicSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetReportsRequest: The batch request containing multiple report
// request.
type GetReportsRequest struct {
	// ReportRequests: Requests, each request will have a separate
	// response.
	// There can be a maximum of 5 requests. All requests should have the
	// same
	// `dateRanges`, `viewId`, `segments`, `samplingLevel`, and
	// `cohortGroup`.
	ReportRequests []*ReportRequest `json:"reportRequests,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReportRequests") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReportRequests") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GetReportsRequest) MarshalJSON() ([]byte, error) {
	type noMethod GetReportsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetReportsResponse: The main response class which holds the reports
// from the Reporting API
// `batchGet` call.
type GetReportsResponse struct {
	// Reports: Responses corresponding to each of the request.
	Reports []*Report `json:"reports,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Reports") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Reports") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GetReportsResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetReportsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Metric:
// [Metrics](https://support.google.com/analytics/answer/1033861)
// are the quantitative measurements. For example, the metric
// `ga:users`
// indicates the total number of users for the requested time period.
type Metric struct {
	// Alias: An alias for the metric expression is an alternate name for
	// the
	// expression. The alias can be used for filtering and sorting. This
	// field
	// is optional and is useful if the expression is not a single metric
	// but
	// a complex expression which cannot be used in filtering and
	// sorting.
	// The alias is also used in the response column header.
	Alias string `json:"alias,omitempty"`

	// Expression: A metric expression in the request. An expression is
	// constructed from one
	// or more metrics and numbers. Accepted operators include: Plus (+),
	// Minus
	// (-), Negation (Unary -), Divided by (/), Multiplied by (*),
	// Parenthesis,
	// Positive cardinal numbers (0-9), can include decimals and is limited
	// to
	// 1024 characters. Example `ga:totalRefunds/ga:users`, in most cases
	// the
	// metric expression is just a single metric name like
	// `ga:users`.
	// Adding mixed `MetricType` (E.g., `CURRENCY` + `PERCENTAGE`)
	// metrics
	// will result in unexpected results.
	Expression string `json:"expression,omitempty"`

	// FormattingType: Specifies how the metric expression should be
	// formatted, for example
	// `INTEGER`.
	//
	// Possible values:
	//   "METRIC_TYPE_UNSPECIFIED" - Metric type is unspecified.
	//   "INTEGER" - Integer metric.
	//   "FLOAT" - Float metric.
	//   "CURRENCY" - Currency metric.
	//   "PERCENT" - Percentage metric.
	//   "TIME" - Time metric in `HH:MM:SS` format.
	FormattingType string `json:"formattingType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Alias") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Alias") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Metric) MarshalJSON() ([]byte, error) {
	type noMethod Metric
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricFilter: MetricFilter specifies the filter on a metric.
type MetricFilter struct {
	// ComparisonValue: The value to compare against.
	ComparisonValue string `json:"comparisonValue,omitempty"`

	// MetricName: The metric that will be filtered on. A metricFilter must
	// contain a metric
	// name. A metric name can be an alias earlier defined as a metric or it
	// can
	// also be a metric expression.
	MetricName string `json:"metricName,omitempty"`

	// Not: Logical `NOT` operator. If this boolean is set to true, then the
	// matching
	// metric values will be excluded in the report. The default is false.
	Not bool `json:"not,omitempty"`

	// Operator: Is the metric `EQUAL`, `LESS_THAN` or `GREATER_THAN`
	// the
	// comparisonValue, the default is `EQUAL`. If the operator
	// is
	// `IS_MISSING`, checks if the metric is missing and would ignore
	// the
	// comparisonValue.
	//
	// Possible values:
	//   "OPERATOR_UNSPECIFIED" - If the operator is not specified, it is
	// treated as `EQUAL`.
	//   "EQUAL" - Should the value of the metric be exactly equal to the
	// comparison value.
	//   "LESS_THAN" - Should the value of the metric be less than to the
	// comparison value.
	//   "GREATER_THAN" - Should the value of the metric be greater than to
	// the comparison value.
	//   "IS_MISSING" - Validates if the metric is missing.
	// Doesn't take comparisonValue into account.
	Operator string `json:"operator,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ComparisonValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ComparisonValue") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MetricFilter) MarshalJSON() ([]byte, error) {
	type noMethod MetricFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricFilterClause: Represents a group of metric filters.
// Set the operator value to specify how the filters are logically
// combined.
type MetricFilterClause struct {
	// Filters: The repeated set of filters. They are logically combined
	// based on the
	// operator specified.
	Filters []*MetricFilter `json:"filters,omitempty"`

	// Operator: The operator for combining multiple metric filters. If
	// unspecified, it is
	// treated as an `OR`.
	//
	// Possible values:
	//   "OPERATOR_UNSPECIFIED" - Unspecified operator. It is treated as an
	// `OR`.
	//   "OR" - The logical `OR` operator.
	//   "AND" - The logical `AND` operator.
	Operator string `json:"operator,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Filters") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricFilterClause) MarshalJSON() ([]byte, error) {
	type noMethod MetricFilterClause
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricHeader: The headers for the metrics.
type MetricHeader struct {
	// MetricHeaderEntries: Headers for the metrics in the response.
	MetricHeaderEntries []*MetricHeaderEntry `json:"metricHeaderEntries,omitempty"`

	// PivotHeaders: Headers for the pivots in the response.
	PivotHeaders []*PivotHeader `json:"pivotHeaders,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MetricHeaderEntries")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MetricHeaderEntries") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MetricHeader) MarshalJSON() ([]byte, error) {
	type noMethod MetricHeader
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricHeaderEntry: Header for the metrics.
type MetricHeaderEntry struct {
	// Name: The name of the header.
	Name string `json:"name,omitempty"`

	// Type: The type of the metric, for example `INTEGER`.
	//
	// Possible values:
	//   "METRIC_TYPE_UNSPECIFIED" - Metric type is unspecified.
	//   "INTEGER" - Integer metric.
	//   "FLOAT" - Float metric.
	//   "CURRENCY" - Currency metric.
	//   "PERCENT" - Percentage metric.
	//   "TIME" - Time metric in `HH:MM:SS` format.
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

func (s *MetricHeaderEntry) MarshalJSON() ([]byte, error) {
	type noMethod MetricHeaderEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OrFiltersForSegment: A list of segment filters in the `OR` group are
// combined with the logical OR
// operator.
type OrFiltersForSegment struct {
	// SegmentFilterClauses: List of segment filters to be combined with a
	// `OR` operator.
	SegmentFilterClauses []*SegmentFilterClause `json:"segmentFilterClauses,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "SegmentFilterClauses") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SegmentFilterClauses") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OrFiltersForSegment) MarshalJSON() ([]byte, error) {
	type noMethod OrFiltersForSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OrderBy: Specifies the sorting options.
type OrderBy struct {
	// FieldName: The field which to sort by. The default sort order is
	// ascending. Example:
	// `ga:browser`.
	// Note, that you can only specify one field for sort here. For
	// example,
	// `ga:browser, ga:city` is not valid.
	FieldName string `json:"fieldName,omitempty"`

	// OrderType: The order type. The default orderType is `VALUE`.
	//
	// Possible values:
	//   "ORDER_TYPE_UNSPECIFIED" - Unspecified order type will be treated
	// as sort based on value.
	//   "VALUE" - The sort order is based on the value of the chosen
	// column; looks only at
	// the first date range.
	//   "DELTA" - The sort order is based on the difference of the values
	// of the chosen
	// column between the first two date ranges.  Usable only if there
	// are
	// exactly two date ranges.
	//   "SMART" - The sort order is based on weighted value of the chosen
	// column.  If
	// column has n/d format, then weighted value of this ratio will
	// be `(n + totals.n)/(d + totals.d)` Usable only for metrics
	// that
	// represent ratios.
	//   "HISTOGRAM_BUCKET" - Histogram order type is applicable only to
	// dimension columns with
	// non-empty histogram-buckets.
	//   "DIMENSION_AS_INTEGER" - If the dimensions are fixed length
	// numbers, ordinary sort would just
	// work fine. `DIMENSION_AS_INTEGER` can be used if the dimensions
	// are
	// variable length numbers.
	OrderType string `json:"orderType,omitempty"`

	// SortOrder: The sorting order for the field.
	//
	// Possible values:
	//   "SORT_ORDER_UNSPECIFIED" - If the sort order is unspecified, the
	// default is ascending.
	//   "ASCENDING" - Ascending sort. The field will be sorted in an
	// ascending manner.
	//   "DESCENDING" - Descending sort. The field will be sorted in a
	// descending manner.
	SortOrder string `json:"sortOrder,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FieldName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FieldName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrderBy) MarshalJSON() ([]byte, error) {
	type noMethod OrderBy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Pivot: The Pivot describes the pivot section in the request.
// The Pivot helps rearrange the information in the table for certain
// reports
// by pivoting your data on a second dimension.
type Pivot struct {
	// DimensionFilterClauses: DimensionFilterClauses are logically combined
	// with an `AND` operator: only
	// data that is included by all these DimensionFilterClauses contributes
	// to
	// the values in this pivot region. Dimension filters can be used to
	// restrict
	// the columns shown in the pivot region. For example if you
	// have
	// `ga:browser` as the requested dimension in the pivot region, and
	// you
	// specify key filters to restrict `ga:browser` to only "IE" or
	// "Firefox",
	// then only those two browsers would show up as columns.
	DimensionFilterClauses []*DimensionFilterClause `json:"dimensionFilterClauses,omitempty"`

	// Dimensions: A list of dimensions to show as pivot columns. A Pivot
	// can have a maximum
	// of 4 dimensions. Pivot dimensions are part of the restriction on
	// the
	// total number of dimensions allowed in the request.
	Dimensions []*Dimension `json:"dimensions,omitempty"`

	// MaxGroupCount: Specifies the maximum number of groups to return.
	// The default value is 10, also the maximum value is 1,000.
	MaxGroupCount int64 `json:"maxGroupCount,omitempty"`

	// Metrics: The pivot metrics. Pivot metrics are part of the
	// restriction on total number of metrics allowed in the request.
	Metrics []*Metric `json:"metrics,omitempty"`

	// StartGroup: If k metrics were requested, then the response will
	// contain some
	// data-dependent multiple of k columns in the report.  E.g., if you
	// pivoted
	// on the dimension `ga:browser` then you'd get k columns for "Firefox",
	// k
	// columns for "IE", k columns for "Chrome", etc. The ordering of the
	// groups
	// of columns is determined by descending order of "total" for the first
	// of
	// the k values.  Ties are broken by lexicographic ordering of the
	// first
	// pivot dimension, then lexicographic ordering of the second
	// pivot
	// dimension, and so on.  E.g., if the totals for the first value
	// for
	// Firefox, IE, and Chrome were 8, 2, 8, respectively, the order of
	// columns
	// would be Chrome, Firefox, IE.
	//
	// The following let you choose which of the groups of k columns
	// are
	// included in the response.
	StartGroup int64 `json:"startGroup,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DimensionFilterClauses") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DimensionFilterClauses")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Pivot) MarshalJSON() ([]byte, error) {
	type noMethod Pivot
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotHeader: The headers for each of the pivot sections defined in
// the request.
type PivotHeader struct {
	// PivotHeaderEntries: A single pivot section header.
	PivotHeaderEntries []*PivotHeaderEntry `json:"pivotHeaderEntries,omitempty"`

	// TotalPivotGroupsCount: The total number of groups for this pivot.
	TotalPivotGroupsCount int64 `json:"totalPivotGroupsCount,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PivotHeaderEntries")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PivotHeaderEntries") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *PivotHeader) MarshalJSON() ([]byte, error) {
	type noMethod PivotHeader
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotHeaderEntry: The headers for the each of the metric column
// corresponding to the metrics
// requested in the pivots section of the response.
type PivotHeaderEntry struct {
	// DimensionNames: The name of the dimensions in the pivot response.
	DimensionNames []string `json:"dimensionNames,omitempty"`

	// DimensionValues: The values for the dimensions in the pivot.
	DimensionValues []string `json:"dimensionValues,omitempty"`

	// Metric: The metric header for the metric in the pivot.
	Metric *MetricHeaderEntry `json:"metric,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DimensionNames") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DimensionNames") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *PivotHeaderEntry) MarshalJSON() ([]byte, error) {
	type noMethod PivotHeaderEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotValueRegion: The metric values in the pivot region.
type PivotValueRegion struct {
	// Values: The values of the metrics in each of the pivot regions.
	Values []string `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Values") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Values") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PivotValueRegion) MarshalJSON() ([]byte, error) {
	type noMethod PivotValueRegion
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Report: The data response corresponding to the request.
type Report struct {
	// ColumnHeader: The column headers.
	ColumnHeader *ColumnHeader `json:"columnHeader,omitempty"`

	// Data: Response data.
	Data *ReportData `json:"data,omitempty"`

	// NextPageToken: Page token to retrieve the next page of results in the
	// list.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnHeader") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnHeader") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Report) MarshalJSON() ([]byte, error) {
	type noMethod Report
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReportData: The data part of the report.
type ReportData struct {
	// DataLastRefreshed: The last time the data in the report was
	// refreshed. All the hits received
	// before this timestamp are included in the calculation of the report.
	DataLastRefreshed string `json:"dataLastRefreshed,omitempty"`

	// IsDataGolden: Indicates if response to this request is golden or not.
	// Data is
	// golden when the exact same request will not produce any new results
	// if
	// asked at a later point in time.
	IsDataGolden bool `json:"isDataGolden,omitempty"`

	// Maximums: Minimum and maximum values seen over all matching rows.
	// These are both
	// empty when `hideValueRanges` in the request is false, or
	// when
	// rowCount is zero.
	Maximums []*DateRangeValues `json:"maximums,omitempty"`

	// Minimums: Minimum and maximum values seen over all matching rows.
	// These are both
	// empty when `hideValueRanges` in the request is false, or
	// when
	// rowCount is zero.
	Minimums []*DateRangeValues `json:"minimums,omitempty"`

	// RowCount: Total number of matching rows for this query.
	RowCount int64 `json:"rowCount,omitempty"`

	// Rows: There's one ReportRow for every unique combination of
	// dimensions.
	Rows []*ReportRow `json:"rows,omitempty"`

	// SamplesReadCounts: If the results
	// are
	// [sampled](https://support.google.com/analytics/answer/2637192),
	// th
	// is returns the total number of samples read, one entry per date
	// range.
	// If the results are not sampled this field will not be defined.
	// See
	// [developer
	// guide](/analytics/devguides/reporting/core/v4/basics#sampling)
	// for details.
	SamplesReadCounts googleapi.Int64s `json:"samplesReadCounts,omitempty"`

	// SamplingSpaceSizes: If the results
	// are
	// [sampled](https://support.google.com/analytics/answer/2637192),
	// th
	// is returns the total number of
	// samples present, one entry per date range. If the results are not
	// sampled
	// this field will not be defined. See
	// [developer
	// guide](/analytics/devguides/reporting/core/v4/basics#sampling)
	// for details.
	SamplingSpaceSizes googleapi.Int64s `json:"samplingSpaceSizes,omitempty"`

	// Totals: For each requested date range, for the set of all rows that
	// match
	// the query, every requested value format gets a total. The total
	// for a value format is computed by first totaling the
	// metrics
	// mentioned in the value format and then evaluating the value
	// format as a scalar expression.  E.g., The "totals" for
	// `3 / (ga:sessions + 2)` we compute
	// `3 / ((sum of all relevant ga:sessions) + 2)`.
	// Totals are computed before pagination.
	Totals []*DateRangeValues `json:"totals,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DataLastRefreshed")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DataLastRefreshed") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ReportData) MarshalJSON() ([]byte, error) {
	type noMethod ReportData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReportRequest: The main request class which specifies the Reporting
// API request.
type ReportRequest struct {
	// CohortGroup: Cohort group associated with this request. If there is a
	// cohort group
	// in the request the `ga:cohort` dimension must be present.
	// Every [ReportRequest](#ReportRequest) within a `batchGet` method
	// must
	// contain the same `cohortGroup` definition.
	CohortGroup *CohortGroup `json:"cohortGroup,omitempty"`

	// DateRanges: Date ranges in the request. The request can have a
	// maximum of 2 date
	// ranges. The response will contain a set of metric values for
	// each
	// combination of the dimensions for each date range in the request. So,
	// if
	// there are two date ranges, there will be two set of metric values,
	// one for
	// the original date range and one for the second date range.
	// The `reportRequest.dateRanges` field should not be specified for
	// cohorts
	// or Lifetime value requests.
	// If a date range is not provided, the default date range is
	// (startDate:
	// current date - 7 days, endDate: current date - 1 day).
	// Every
	// [ReportRequest](#ReportRequest) within a `batchGet` method
	// must
	// contain the same `dateRanges` definition.
	DateRanges []*DateRange `json:"dateRanges,omitempty"`

	// DimensionFilterClauses: The dimension filter clauses for filtering
	// Dimension Values. They are
	// logically combined with the `AND` operator. Note that filtering
	// occurs
	// before any dimensions are aggregated, so that the returned
	// metrics
	// represent the total for only the relevant dimensions.
	DimensionFilterClauses []*DimensionFilterClause `json:"dimensionFilterClauses,omitempty"`

	// Dimensions: The dimensions requested.
	// Requests can have a total of 7 dimensions.
	Dimensions []*Dimension `json:"dimensions,omitempty"`

	// FiltersExpression: Dimension or metric filters that restrict the data
	// returned for your
	// request. To use the `filtersExpression`, supply a dimension or metric
	// on
	// which to filter, followed by the filter expression. For example,
	// the
	// following expression selects `ga:browser` dimension which starts
	// with
	// Firefox; `ga:browser=~^Firefox`. For more information on
	// dimensions
	// and metric filters, see
	// [Filters
	// reference](https://developers.google.com/analytics/devguides/reporting
	// /core/v3/reference#filters).
	FiltersExpression string `json:"filtersExpression,omitempty"`

	// HideTotals: If set to true, hides the total of all metrics for all
	// the matching rows,
	// for every date range. The default false and will return the totals.
	HideTotals bool `json:"hideTotals,omitempty"`

	// HideValueRanges: If set to true, hides the minimum and maximum across
	// all matching rows.
	// The default is false and the value ranges are returned.
	HideValueRanges bool `json:"hideValueRanges,omitempty"`

	// IncludeEmptyRows: If set to false, the response does not include rows
	// if all the retrieved
	// metrics are equal to zero. The default is false which will exclude
	// these
	// rows.
	IncludeEmptyRows bool `json:"includeEmptyRows,omitempty"`

	// MetricFilterClauses: The metric filter clauses. They are logically
	// combined with the `AND`
	// operator.  Metric filters look at only the first date range and not
	// the
	// comparing date range. Note that filtering on metrics occurs after
	// the
	// metrics are aggregated.
	MetricFilterClauses []*MetricFilterClause `json:"metricFilterClauses,omitempty"`

	// Metrics: The metrics requested.
	// Requests must specify at least one metric. Requests can have a
	// total of 10 metrics.
	Metrics []*Metric `json:"metrics,omitempty"`

	// OrderBys: Sort order on output rows. To compare two rows, the
	// elements of the
	// following are applied in order until a difference is found.  All
	// date
	// ranges in the output get the same row order.
	OrderBys []*OrderBy `json:"orderBys,omitempty"`

	// PageSize: Page size is for paging and specifies the maximum number of
	// returned rows.
	// Page size should be >= 0. A query returns the default of 1,000
	// rows.
	// The Analytics Core Reporting API returns a maximum of 10,000 rows
	// per
	// request, no matter how many you ask for. It can also return fewer
	// rows
	// than requested, if there aren't as many dimension segments as you
	// expect.
	// For instance, there are fewer than 300 possible values for
	// `ga:country`,
	// so when segmenting only by country, you can't get more than 300
	// rows,
	// even if you set `pageSize` to a higher value.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: A continuation token to get the next page of the results.
	// Adding this to
	// the request will return the rows after the pageToken. The pageToken
	// should
	// be the value returned in the nextPageToken parameter in the response
	// to
	// the GetReports request.
	PageToken string `json:"pageToken,omitempty"`

	// Pivots: The pivot definitions. Requests can have a maximum of 2
	// pivots.
	Pivots []*Pivot `json:"pivots,omitempty"`

	// SamplingLevel: The desired
	// report
	// [sample](https://support.google.com/analytics/answer/2637192)
	// size.
	// If the the `samplingLevel` field is unspecified the `DEFAULT`
	// sampling
	// level is used. Every [ReportRequest](#ReportRequest) within
	// a
	// `batchGet` method must contain the same `samplingLevel` definition.
	// See
	// [developer
	// guide](/analytics/devguides/reporting/core/v4/basics#sampling)
	//  for details.
	//
	// Possible values:
	//   "SAMPLING_UNSPECIFIED" - If the `samplingLevel` field is
	// unspecified the `DEFAULT` sampling level
	// is used.
	//   "DEFAULT" - Returns response with a sample size that balances speed
	// and
	// accuracy.
	//   "SMALL" - It returns a fast response with a smaller sampling size.
	//   "LARGE" - Returns a more accurate response using a large sampling
	// size. But this
	// may result in response being slower.
	SamplingLevel string `json:"samplingLevel,omitempty"`

	// Segments: Segment the data returned for the request. A segment
	// definition helps look
	// at a subset of the segment request. A request can contain up to
	// four
	// segments. Every [ReportRequest](#ReportRequest) within a
	// `batchGet` method must contain the same `segments` definition.
	// Requests
	// with segments must have the `ga:segment` dimension.
	Segments []*Segment `json:"segments,omitempty"`

	// ViewId: The Analytics
	// [view ID](https://support.google.com/analytics/answer/1009618)
	// from which to retrieve data. Every
	// [ReportRequest](#ReportRequest)
	// within a `batchGet` method must contain the same `viewId`.
	ViewId string `json:"viewId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CohortGroup") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CohortGroup") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReportRequest) MarshalJSON() ([]byte, error) {
	type noMethod ReportRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReportRow: A row in the report.
type ReportRow struct {
	// Dimensions: List of requested dimensions.
	Dimensions []string `json:"dimensions,omitempty"`

	// Metrics: List of metrics for each requested DateRange.
	Metrics []*DateRangeValues `json:"metrics,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Dimensions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Dimensions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReportRow) MarshalJSON() ([]byte, error) {
	type noMethod ReportRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Segment: The segment definition, if the report needs to be
// segmented.
// A Segment is a subset of the Analytics data. For example, of the
// entire
// set of users, one Segment might be users from a particular country or
// city.
type Segment struct {
	// DynamicSegment: A dynamic segment definition in the request.
	DynamicSegment *DynamicSegment `json:"dynamicSegment,omitempty"`

	// SegmentId: The segment ID of a built-in or custom segment, for
	// example `gaid::-3`.
	SegmentId string `json:"segmentId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DynamicSegment") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DynamicSegment") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Segment) MarshalJSON() ([]byte, error) {
	type noMethod Segment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentDefinition: SegmentDefinition defines the segment to be a set
// of SegmentFilters which
// are combined together with a logical `AND` operation.
type SegmentDefinition struct {
	// SegmentFilters: A segment is defined by a set of segment filters
	// which are combined
	// together with a logical `AND` operation.
	SegmentFilters []*SegmentFilter `json:"segmentFilters,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SegmentFilters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SegmentFilters") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SegmentDefinition) MarshalJSON() ([]byte, error) {
	type noMethod SegmentDefinition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentDimensionFilter: Dimension filter specifies the filtering
// options on a dimension.
type SegmentDimensionFilter struct {
	// CaseSensitive: Should the match be case sensitive, ignored for
	// `IN_LIST` operator.
	CaseSensitive bool `json:"caseSensitive,omitempty"`

	// DimensionName: Name of the dimension for which the filter is being
	// applied.
	DimensionName string `json:"dimensionName,omitempty"`

	// Expressions: The list of expressions, only the first element is used
	// for all operators
	Expressions []string `json:"expressions,omitempty"`

	// MaxComparisonValue: Maximum comparison values for `BETWEEN` match
	// type.
	MaxComparisonValue string `json:"maxComparisonValue,omitempty"`

	// MinComparisonValue: Minimum comparison values for `BETWEEN` match
	// type.
	MinComparisonValue string `json:"minComparisonValue,omitempty"`

	// Operator: The operator to use to match the dimension with the
	// expressions.
	//
	// Possible values:
	//   "OPERATOR_UNSPECIFIED" - If the match type is unspecified, it is
	// treated as a REGEXP.
	//   "REGEXP" - The match expression is treated as a regular expression.
	// All other match
	// types are not treated as regular expressions.
	//   "BEGINS_WITH" - Matches the values which begin with the match
	// expression provided.
	//   "ENDS_WITH" - Matches the values which end with the match
	// expression provided.
	//   "PARTIAL" - Substring match.
	//   "EXACT" - The value should match the match expression entirely.
	//   "IN_LIST" - This option is used to specify a dimension filter whose
	// expression can
	// take any value from a selected list of values. This helps
	// avoiding
	// evaluating multiple exact match dimension filters which are OR'ed
	// for
	// every single response row. For example:
	//
	//     expressions: ["A", "B", "C"]
	//
	// Any response row whose dimension has it is value as A, B or C,
	// matches
	// this DimensionFilter.
	//   "NUMERIC_LESS_THAN" - Integer comparison filters.
	// case sensitivity is ignored for these and the expression
	// is assumed to be a string representing an integer.
	// Failure conditions:
	//
	// - if expression is not a valid int64, the client should expect
	//   an error.
	// - input dimensions that are not valid int64 values will never match
	// the
	//   filter.
	//
	// Checks if the dimension is numerically less than the match
	// expression.
	//   "NUMERIC_GREATER_THAN" - Checks if the dimension is numerically
	// greater than the match
	// expression.
	//   "NUMERIC_BETWEEN" - Checks if the dimension is numerically between
	// the minimum and maximum
	// of the match expression, boundaries excluded.
	Operator string `json:"operator,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CaseSensitive") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CaseSensitive") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SegmentDimensionFilter) MarshalJSON() ([]byte, error) {
	type noMethod SegmentDimensionFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentFilter: SegmentFilter defines the segment to be either a
// simple or a sequence
// segment. A simple segment condition contains dimension and metric
// conditions
// to select the sessions or users. A sequence segment condition can be
// used to
// select users or sessions based on sequential conditions.
type SegmentFilter struct {
	// Not: If true, match the complement of simple or sequence segment.
	// For example, to match all visits not from "New York", we can define
	// the
	// segment as follows:
	//
	//       "sessionSegment": {
	//         "segmentFilters": [{
	//           "simpleSegment" :{
	//             "orFiltersForSegment": [{
	//               "segmentFilterClauses":[{
	//                 "dimensionFilter": {
	//                   "dimensionName": "ga:city",
	//                   "expressions": ["New York"]
	//                 }
	//               }]
	//             }]
	//           },
	//           "not": "True"
	//         }]
	//       },
	Not bool `json:"not,omitempty"`

	// SequenceSegment: Sequence conditions consist of one or more steps,
	// where each step is
	// defined by one or more dimension/metric conditions. Multiple steps
	// can
	// be combined with special sequence operators.
	SequenceSegment *SequenceSegment `json:"sequenceSegment,omitempty"`

	// SimpleSegment: A Simple segment conditions consist of one or more
	// dimension/metric
	// conditions that can be combined
	SimpleSegment *SimpleSegment `json:"simpleSegment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Not") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Not") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SegmentFilter) MarshalJSON() ([]byte, error) {
	type noMethod SegmentFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentFilterClause: Filter Clause to be used in a segment
// definition, can be wither a metric or
// a dimension filter.
type SegmentFilterClause struct {
	// DimensionFilter: Dimension Filter for the segment definition.
	DimensionFilter *SegmentDimensionFilter `json:"dimensionFilter,omitempty"`

	// MetricFilter: Metric Filter for the segment definition.
	MetricFilter *SegmentMetricFilter `json:"metricFilter,omitempty"`

	// Not: Matches the complement (`!`) of the filter.
	Not bool `json:"not,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DimensionFilter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DimensionFilter") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SegmentFilterClause) MarshalJSON() ([]byte, error) {
	type noMethod SegmentFilterClause
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentMetricFilter: Metric filter to be used in a segment filter
// clause.
type SegmentMetricFilter struct {
	// ComparisonValue: The value to compare against. If the operator is
	// `BETWEEN`, this value is
	// treated as minimum comparison value.
	ComparisonValue string `json:"comparisonValue,omitempty"`

	// MaxComparisonValue: Max comparison value is only used for `BETWEEN`
	// operator.
	MaxComparisonValue string `json:"maxComparisonValue,omitempty"`

	// MetricName: The metric that will be filtered on. A `metricFilter`
	// must contain a
	// metric name.
	MetricName string `json:"metricName,omitempty"`

	// Operator: Specifies is the operation to perform to compare the
	// metric. The default
	// is `EQUAL`.
	//
	// Possible values:
	//   "UNSPECIFIED_OPERATOR" - Unspecified operator is treated as
	// `LESS_THAN` operator.
	//   "LESS_THAN" - Checks if the metric value is less than comparison
	// value.
	//   "GREATER_THAN" - Checks if the metric value is greater than
	// comparison value.
	//   "EQUAL" - Equals operator.
	//   "BETWEEN" - For between operator, both the minimum and maximum are
	// exclusive.
	// We will use `LT` and `GT` for comparison.
	Operator string `json:"operator,omitempty"`

	// Scope: Scope for a metric defines the level at which that metric is
	// defined.  The
	// specified metric scope must be equal to or greater than its primary
	// scope
	// as defined in the data model. The primary scope is defined by if
	// the
	// segment is selecting users or sessions.
	//
	// Possible values:
	//   "UNSPECIFIED_SCOPE" - If the scope is unspecified, it defaults to
	// the condition scope,
	// `USER` or `SESSION` depending on if the segment is trying to
	// choose
	// users or sessions.
	//   "PRODUCT" - Product scope.
	//   "HIT" - Hit scope.
	//   "SESSION" - Session scope.
	//   "USER" - User scope.
	Scope string `json:"scope,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ComparisonValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ComparisonValue") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SegmentMetricFilter) MarshalJSON() ([]byte, error) {
	type noMethod SegmentMetricFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentSequenceStep: A segment sequence definition.
type SegmentSequenceStep struct {
	// MatchType: Specifies if the step immediately precedes or can be any
	// time before the
	// next step.
	//
	// Possible values:
	//   "UNSPECIFIED_MATCH_TYPE" - Unspecified match type is treated as
	// precedes.
	//   "PRECEDES" - Operator indicates that the previous step precedes the
	// next step.
	//   "IMMEDIATELY_PRECEDES" - Operator indicates that the previous step
	// immediately precedes the next
	// step.
	MatchType string `json:"matchType,omitempty"`

	// OrFiltersForSegment: A sequence is specified with a list of Or
	// grouped filters which are
	// combined with `AND` operator.
	OrFiltersForSegment []*OrFiltersForSegment `json:"orFiltersForSegment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MatchType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MatchType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SegmentSequenceStep) MarshalJSON() ([]byte, error) {
	type noMethod SegmentSequenceStep
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SequenceSegment: Sequence conditions consist of one or more steps,
// where each step is defined
// by one or more dimension/metric conditions. Multiple steps can be
// combined
// with special sequence operators.
type SequenceSegment struct {
	// FirstStepShouldMatchFirstHit: If set, first step condition must match
	// the first hit of the visitor (in
	// the date range).
	FirstStepShouldMatchFirstHit bool `json:"firstStepShouldMatchFirstHit,omitempty"`

	// SegmentSequenceSteps: The list of steps in the sequence.
	SegmentSequenceSteps []*SegmentSequenceStep `json:"segmentSequenceSteps,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "FirstStepShouldMatchFirstHit") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "FirstStepShouldMatchFirstHit") to include in API requests with the
	// JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SequenceSegment) MarshalJSON() ([]byte, error) {
	type noMethod SequenceSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SimpleSegment: A Simple segment conditions consist of one or more
// dimension/metric
// conditions that can be combined.
type SimpleSegment struct {
	// OrFiltersForSegment: A list of segment filters groups which are
	// combined with logical `AND`
	// operator.
	OrFiltersForSegment []*OrFiltersForSegment `json:"orFiltersForSegment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OrFiltersForSegment")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OrFiltersForSegment") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SimpleSegment) MarshalJSON() ([]byte, error) {
	type noMethod SimpleSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "analyticsreporting.reports.batchGet":

type ReportsBatchGetCall struct {
	s                 *Service
	getreportsrequest *GetReportsRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// BatchGet: Returns the Analytics data.
func (r *ReportsService) BatchGet(getreportsrequest *GetReportsRequest) *ReportsBatchGetCall {
	c := &ReportsBatchGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.getreportsrequest = getreportsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsBatchGetCall) Fields(s ...googleapi.Field) *ReportsBatchGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsBatchGetCall) Context(ctx context.Context) *ReportsBatchGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReportsBatchGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReportsBatchGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.getreportsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/reports:batchGet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "analyticsreporting.reports.batchGet" call.
// Exactly one of *GetReportsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetReportsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReportsBatchGetCall) Do(opts ...googleapi.CallOption) (*GetReportsResponse, error) {
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
	ret := &GetReportsResponse{
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
	//   "description": "Returns the Analytics data.",
	//   "flatPath": "v4/reports:batchGet",
	//   "httpMethod": "POST",
	//   "id": "analyticsreporting.reports.batchGet",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v4/reports:batchGet",
	//   "request": {
	//     "$ref": "GetReportsRequest"
	//   },
	//   "response": {
	//     "$ref": "GetReportsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}
