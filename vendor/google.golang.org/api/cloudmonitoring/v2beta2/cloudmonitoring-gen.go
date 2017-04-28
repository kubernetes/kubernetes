// Package cloudmonitoring provides access to the Cloud Monitoring API.
//
// See https://cloud.google.com/monitoring/v2beta2/
//
// Usage example:
//
//   import "google.golang.org/api/cloudmonitoring/v2beta2"
//   ...
//   cloudmonitoringService, err := cloudmonitoring.New(oauthHttpClient)
package cloudmonitoring

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

const apiId = "cloudmonitoring:v2beta2"
const apiName = "cloudmonitoring"
const apiVersion = "v2beta2"
const basePath = "https://www.googleapis.com/cloudmonitoring/v2beta2/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View and write monitoring data for all of your Google and third-party
	// Cloud and API projects
	MonitoringScope = "https://www.googleapis.com/auth/monitoring"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.MetricDescriptors = NewMetricDescriptorsService(s)
	s.Timeseries = NewTimeseriesService(s)
	s.TimeseriesDescriptors = NewTimeseriesDescriptorsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	MetricDescriptors *MetricDescriptorsService

	Timeseries *TimeseriesService

	TimeseriesDescriptors *TimeseriesDescriptorsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewMetricDescriptorsService(s *Service) *MetricDescriptorsService {
	rs := &MetricDescriptorsService{s: s}
	return rs
}

type MetricDescriptorsService struct {
	s *Service
}

func NewTimeseriesService(s *Service) *TimeseriesService {
	rs := &TimeseriesService{s: s}
	return rs
}

type TimeseriesService struct {
	s *Service
}

func NewTimeseriesDescriptorsService(s *Service) *TimeseriesDescriptorsService {
	rs := &TimeseriesDescriptorsService{s: s}
	return rs
}

type TimeseriesDescriptorsService struct {
	s *Service
}

// DeleteMetricDescriptorResponse: The response of
// cloudmonitoring.metricDescriptors.delete.
type DeleteMetricDescriptorResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#deleteMetricDescriptorResponse".
	Kind string `json:"kind,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteMetricDescriptorResponse) MarshalJSON() ([]byte, error) {
	type noMethod DeleteMetricDescriptorResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListMetricDescriptorsRequest: The request of
// cloudmonitoring.metricDescriptors.list.
type ListMetricDescriptorsRequest struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#listMetricDescriptorsRequest".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListMetricDescriptorsRequest) MarshalJSON() ([]byte, error) {
	type noMethod ListMetricDescriptorsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListMetricDescriptorsResponse: The response of
// cloudmonitoring.metricDescriptors.list.
type ListMetricDescriptorsResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#listMetricDescriptorsResponse".
	Kind string `json:"kind,omitempty"`

	// Metrics: The returned metric descriptors.
	Metrics []*MetricDescriptor `json:"metrics,omitempty"`

	// NextPageToken: Pagination token. If present, indicates that
	// additional results are available for retrieval. To access the results
	// past the pagination limit, pass this value to the pageToken query
	// parameter.
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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListMetricDescriptorsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListMetricDescriptorsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTimeseriesDescriptorsRequest: The request of
// cloudmonitoring.timeseriesDescriptors.list
type ListTimeseriesDescriptorsRequest struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#listTimeseriesDescriptorsRequest".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListTimeseriesDescriptorsRequest) MarshalJSON() ([]byte, error) {
	type noMethod ListTimeseriesDescriptorsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTimeseriesDescriptorsResponse: The response of
// cloudmonitoring.timeseriesDescriptors.list
type ListTimeseriesDescriptorsResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#listTimeseriesDescriptorsResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Pagination token. If present, indicates that
	// additional results are available for retrieval. To access the results
	// past the pagination limit, set this value to the pageToken query
	// parameter.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Oldest: The oldest timestamp of the interval of this query, as an RFC
	// 3339 string.
	Oldest string `json:"oldest,omitempty"`

	// Timeseries: The returned time series descriptors.
	Timeseries []*TimeseriesDescriptor `json:"timeseries,omitempty"`

	// Youngest: The youngest timestamp of the interval of this query, as an
	// RFC 3339 string.
	Youngest string `json:"youngest,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListTimeseriesDescriptorsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTimeseriesDescriptorsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTimeseriesRequest: The request of cloudmonitoring.timeseries.list
type ListTimeseriesRequest struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#listTimeseriesRequest".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListTimeseriesRequest) MarshalJSON() ([]byte, error) {
	type noMethod ListTimeseriesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTimeseriesResponse: The response of
// cloudmonitoring.timeseries.list
type ListTimeseriesResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#listTimeseriesResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Pagination token. If present, indicates that
	// additional results are available for retrieval. To access the results
	// past the pagination limit, set the pageToken query parameter to this
	// value. All of the points of a time series will be returned before
	// returning any point of the subsequent time series.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Oldest: The oldest timestamp of the interval of this query as an RFC
	// 3339 string.
	Oldest string `json:"oldest,omitempty"`

	// Timeseries: The returned time series.
	Timeseries []*Timeseries `json:"timeseries,omitempty"`

	// Youngest: The youngest timestamp of the interval of this query as an
	// RFC 3339 string.
	Youngest string `json:"youngest,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListTimeseriesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTimeseriesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricDescriptor: A metricDescriptor defines the name, label keys,
// and data type of a particular metric.
type MetricDescriptor struct {
	// Description: Description of this metric.
	Description string `json:"description,omitempty"`

	// Labels: Labels defined for this metric.
	Labels []*MetricDescriptorLabelDescriptor `json:"labels,omitempty"`

	// Name: The name of this metric.
	Name string `json:"name,omitempty"`

	// Project: The project ID to which the metric belongs.
	Project string `json:"project,omitempty"`

	// TypeDescriptor: Type description for this metric.
	TypeDescriptor *MetricDescriptorTypeDescriptor `json:"typeDescriptor,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Description") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod MetricDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricDescriptorLabelDescriptor: A label in a metric is a description
// of this metric, including the key of this description (what the
// description is), and the value for this description.
type MetricDescriptorLabelDescriptor struct {
	// Description: Label description.
	Description string `json:"description,omitempty"`

	// Key: Label key.
	Key string `json:"key,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Description") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricDescriptorLabelDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod MetricDescriptorLabelDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricDescriptorTypeDescriptor: A type in a metric contains
// information about how the metric is collected and what its data
// points look like.
type MetricDescriptorTypeDescriptor struct {
	// MetricType: The method of collecting data for the metric. See Metric
	// types.
	MetricType string `json:"metricType,omitempty"`

	// ValueType: The data type of of individual points in the metric's time
	// series. See Metric value types.
	ValueType string `json:"valueType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MetricType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MetricType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricDescriptorTypeDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod MetricDescriptorTypeDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Point: Point is a single point in a time series. It consists of a
// start time, an end time, and a value.
type Point struct {
	// BoolValue: The value of this data point. Either "true" or "false".
	BoolValue *bool `json:"boolValue,omitempty"`

	// DistributionValue: The value of this data point as a distribution. A
	// distribution value can contain a list of buckets and/or an
	// underflowBucket and an overflowBucket. The values of these points can
	// be used to create a histogram.
	DistributionValue *PointDistribution `json:"distributionValue,omitempty"`

	// DoubleValue: The value of this data point as a double-precision
	// floating-point number.
	DoubleValue *float64 `json:"doubleValue,omitempty"`

	// End: The interval [start, end] is the time period to which the
	// point's value applies. For gauge metrics, whose values are
	// instantaneous measurements, this interval should be empty (start
	// should equal end). For cumulative metrics (of which deltas and rates
	// are special cases), the interval should be non-empty. Both start and
	// end are RFC 3339 strings.
	End string `json:"end,omitempty"`

	// Int64Value: The value of this data point as a 64-bit integer.
	Int64Value *int64 `json:"int64Value,omitempty,string"`

	// Start: The interval [start, end] is the time period to which the
	// point's value applies. For gauge metrics, whose values are
	// instantaneous measurements, this interval should be empty (start
	// should equal end). For cumulative metrics (of which deltas and rates
	// are special cases), the interval should be non-empty. Both start and
	// end are RFC 3339 strings.
	Start string `json:"start,omitempty"`

	// StringValue: The value of this data point in string format.
	StringValue *string `json:"stringValue,omitempty"`

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

func (s *Point) MarshalJSON() ([]byte, error) {
	type noMethod Point
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Point) UnmarshalJSON(data []byte) error {
	type noMethod Point
	var s1 struct {
		DoubleValue *gensupport.JSONFloat64 `json:"doubleValue"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	if s1.DoubleValue != nil {
		s.DoubleValue = (*float64)(s1.DoubleValue)
	}
	return nil
}

// PointDistribution: Distribution data point value type. When writing
// distribution points, try to be consistent with the boundaries of your
// buckets. If you must modify the bucket boundaries, then do so by
// merging, partitioning, or appending rather than skewing them.
type PointDistribution struct {
	// Buckets: The finite buckets.
	Buckets []*PointDistributionBucket `json:"buckets,omitempty"`

	// OverflowBucket: The overflow bucket.
	OverflowBucket *PointDistributionOverflowBucket `json:"overflowBucket,omitempty"`

	// UnderflowBucket: The underflow bucket.
	UnderflowBucket *PointDistributionUnderflowBucket `json:"underflowBucket,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Buckets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Buckets") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PointDistribution) MarshalJSON() ([]byte, error) {
	type noMethod PointDistribution
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PointDistributionBucket: The histogram's bucket. Buckets that form
// the histogram of a distribution value. If the upper bound of a
// bucket, say U1, does not equal the lower bound of the next bucket,
// say L2, this means that there is no event in [U1, L2).
type PointDistributionBucket struct {
	// Count: The number of events whose values are in the interval defined
	// by this bucket.
	Count int64 `json:"count,omitempty,string"`

	// LowerBound: The lower bound of the value interval of this bucket
	// (inclusive).
	LowerBound float64 `json:"lowerBound,omitempty"`

	// UpperBound: The upper bound of the value interval of this bucket
	// (exclusive).
	UpperBound float64 `json:"upperBound,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Count") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PointDistributionBucket) MarshalJSON() ([]byte, error) {
	type noMethod PointDistributionBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *PointDistributionBucket) UnmarshalJSON(data []byte) error {
	type noMethod PointDistributionBucket
	var s1 struct {
		LowerBound gensupport.JSONFloat64 `json:"lowerBound"`
		UpperBound gensupport.JSONFloat64 `json:"upperBound"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.LowerBound = float64(s1.LowerBound)
	s.UpperBound = float64(s1.UpperBound)
	return nil
}

// PointDistributionOverflowBucket: The overflow bucket is a special
// bucket that does not have the upperBound field; it includes all of
// the events that are no less than its lower bound.
type PointDistributionOverflowBucket struct {
	// Count: The number of events whose values are in the interval defined
	// by this bucket.
	Count int64 `json:"count,omitempty,string"`

	// LowerBound: The lower bound of the value interval of this bucket
	// (inclusive).
	LowerBound float64 `json:"lowerBound,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Count") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PointDistributionOverflowBucket) MarshalJSON() ([]byte, error) {
	type noMethod PointDistributionOverflowBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *PointDistributionOverflowBucket) UnmarshalJSON(data []byte) error {
	type noMethod PointDistributionOverflowBucket
	var s1 struct {
		LowerBound gensupport.JSONFloat64 `json:"lowerBound"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.LowerBound = float64(s1.LowerBound)
	return nil
}

// PointDistributionUnderflowBucket: The underflow bucket is a special
// bucket that does not have the lowerBound field; it includes all of
// the events that are less than its upper bound.
type PointDistributionUnderflowBucket struct {
	// Count: The number of events whose values are in the interval defined
	// by this bucket.
	Count int64 `json:"count,omitempty,string"`

	// UpperBound: The upper bound of the value interval of this bucket
	// (exclusive).
	UpperBound float64 `json:"upperBound,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Count") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PointDistributionUnderflowBucket) MarshalJSON() ([]byte, error) {
	type noMethod PointDistributionUnderflowBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *PointDistributionUnderflowBucket) UnmarshalJSON(data []byte) error {
	type noMethod PointDistributionUnderflowBucket
	var s1 struct {
		UpperBound gensupport.JSONFloat64 `json:"upperBound"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.UpperBound = float64(s1.UpperBound)
	return nil
}

// Timeseries: The monitoring data is organized as metrics and stored as
// data points that are recorded over time. Each data point represents
// information like the CPU utilization of your virtual machine. A
// historical record of these data points is called a time series.
type Timeseries struct {
	// Points: The data points of this time series. The points are listed in
	// order of their end timestamp, from younger to older.
	Points []*Point `json:"points,omitempty"`

	// TimeseriesDesc: The descriptor of this time series.
	TimeseriesDesc *TimeseriesDescriptor `json:"timeseriesDesc,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Points") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Points") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Timeseries) MarshalJSON() ([]byte, error) {
	type noMethod Timeseries
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TimeseriesDescriptor: TimeseriesDescriptor identifies a single time
// series.
type TimeseriesDescriptor struct {
	// Labels: The label's name.
	Labels map[string]string `json:"labels,omitempty"`

	// Metric: The name of the metric.
	Metric string `json:"metric,omitempty"`

	// Project: The Developers Console project number to which this time
	// series belongs.
	Project string `json:"project,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Labels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Labels") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TimeseriesDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod TimeseriesDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TimeseriesDescriptorLabel struct {
	// Key: The label's name.
	Key string `json:"key,omitempty"`

	// Value: The label's value.
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

func (s *TimeseriesDescriptorLabel) MarshalJSON() ([]byte, error) {
	type noMethod TimeseriesDescriptorLabel
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TimeseriesPoint: When writing time series, TimeseriesPoint should be
// used instead of Timeseries, to enforce single point for each time
// series in the timeseries.write request.
type TimeseriesPoint struct {
	// Point: The data point in this time series snapshot.
	Point *Point `json:"point,omitempty"`

	// TimeseriesDesc: The descriptor of this time series.
	TimeseriesDesc *TimeseriesDescriptor `json:"timeseriesDesc,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Point") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Point") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TimeseriesPoint) MarshalJSON() ([]byte, error) {
	type noMethod TimeseriesPoint
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WriteTimeseriesRequest: The request of
// cloudmonitoring.timeseries.write
type WriteTimeseriesRequest struct {
	// CommonLabels: The label's name.
	CommonLabels map[string]string `json:"commonLabels,omitempty"`

	// Timeseries: Provide time series specific labels and the data points
	// for each time series. The labels in timeseries and the common_labels
	// should form a complete list of labels that required by the metric.
	Timeseries []*TimeseriesPoint `json:"timeseries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommonLabels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CommonLabels") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WriteTimeseriesRequest) MarshalJSON() ([]byte, error) {
	type noMethod WriteTimeseriesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WriteTimeseriesResponse: The response of
// cloudmonitoring.timeseries.write
type WriteTimeseriesResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "cloudmonitoring#writeTimeseriesResponse".
	Kind string `json:"kind,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WriteTimeseriesResponse) MarshalJSON() ([]byte, error) {
	type noMethod WriteTimeseriesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "cloudmonitoring.metricDescriptors.create":

type MetricDescriptorsCreateCall struct {
	s                *Service
	project          string
	metricdescriptor *MetricDescriptor
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Create: Create a new metric.
func (r *MetricDescriptorsService) Create(project string, metricdescriptor *MetricDescriptor) *MetricDescriptorsCreateCall {
	c := &MetricDescriptorsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.metricdescriptor = metricdescriptor
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MetricDescriptorsCreateCall) Fields(s ...googleapi.Field) *MetricDescriptorsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MetricDescriptorsCreateCall) Context(ctx context.Context) *MetricDescriptorsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *MetricDescriptorsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *MetricDescriptorsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.metricdescriptor)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/metricDescriptors")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudmonitoring.metricDescriptors.create" call.
// Exactly one of *MetricDescriptor or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *MetricDescriptor.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MetricDescriptorsCreateCall) Do(opts ...googleapi.CallOption) (*MetricDescriptor, error) {
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
	ret := &MetricDescriptor{
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
	//   "description": "Create a new metric.",
	//   "httpMethod": "POST",
	//   "id": "cloudmonitoring.metricDescriptors.create",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project id. The value can be the numeric project ID or string-based project name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/metricDescriptors",
	//   "request": {
	//     "$ref": "MetricDescriptor"
	//   },
	//   "response": {
	//     "$ref": "MetricDescriptor"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/monitoring"
	//   ]
	// }

}

// method id "cloudmonitoring.metricDescriptors.delete":

type MetricDescriptorsDeleteCall struct {
	s          *Service
	project    string
	metric     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Delete an existing metric.
func (r *MetricDescriptorsService) Delete(project string, metric string) *MetricDescriptorsDeleteCall {
	c := &MetricDescriptorsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.metric = metric
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MetricDescriptorsDeleteCall) Fields(s ...googleapi.Field) *MetricDescriptorsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MetricDescriptorsDeleteCall) Context(ctx context.Context) *MetricDescriptorsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *MetricDescriptorsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *MetricDescriptorsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/metricDescriptors/{metric}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"metric":  c.metric,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudmonitoring.metricDescriptors.delete" call.
// Exactly one of *DeleteMetricDescriptorResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *DeleteMetricDescriptorResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MetricDescriptorsDeleteCall) Do(opts ...googleapi.CallOption) (*DeleteMetricDescriptorResponse, error) {
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
	ret := &DeleteMetricDescriptorResponse{
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
	//   "description": "Delete an existing metric.",
	//   "httpMethod": "DELETE",
	//   "id": "cloudmonitoring.metricDescriptors.delete",
	//   "parameterOrder": [
	//     "project",
	//     "metric"
	//   ],
	//   "parameters": {
	//     "metric": {
	//       "description": "Name of the metric.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID to which the metric belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/metricDescriptors/{metric}",
	//   "response": {
	//     "$ref": "DeleteMetricDescriptorResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/monitoring"
	//   ]
	// }

}

// method id "cloudmonitoring.metricDescriptors.list":

type MetricDescriptorsListCall struct {
	s                            *Service
	project                      string
	listmetricdescriptorsrequest *ListMetricDescriptorsRequest
	urlParams_                   gensupport.URLParams
	ifNoneMatch_                 string
	ctx_                         context.Context
	header_                      http.Header
}

// List: List metric descriptors that match the query. If the query is
// not set, then all of the metric descriptors will be returned. Large
// responses will be paginated, use the nextPageToken returned in the
// response to request subsequent pages of results by setting the
// pageToken query parameter to the value of the nextPageToken.
func (r *MetricDescriptorsService) List(project string, listmetricdescriptorsrequest *ListMetricDescriptorsRequest) *MetricDescriptorsListCall {
	c := &MetricDescriptorsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.listmetricdescriptorsrequest = listmetricdescriptorsrequest
	return c
}

// Count sets the optional parameter "count": Maximum number of metric
// descriptors per page. Used for pagination. If not specified, count =
// 100.
func (c *MetricDescriptorsListCall) Count(count int64) *MetricDescriptorsListCall {
	c.urlParams_.Set("count", fmt.Sprint(count))
	return c
}

// PageToken sets the optional parameter "pageToken": The pagination
// token, which is used to page through large result sets. Set this
// value to the value of the nextPageToken to retrieve the next page of
// results.
func (c *MetricDescriptorsListCall) PageToken(pageToken string) *MetricDescriptorsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Query sets the optional parameter "query": The query used to search
// against existing metrics. Separate keywords with a space; the service
// joins all keywords with AND, meaning that all keywords must match for
// a metric to be returned. If this field is omitted, all metrics are
// returned. If an empty string is passed with this field, no metrics
// are returned.
func (c *MetricDescriptorsListCall) Query(query string) *MetricDescriptorsListCall {
	c.urlParams_.Set("query", query)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MetricDescriptorsListCall) Fields(s ...googleapi.Field) *MetricDescriptorsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MetricDescriptorsListCall) IfNoneMatch(entityTag string) *MetricDescriptorsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MetricDescriptorsListCall) Context(ctx context.Context) *MetricDescriptorsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *MetricDescriptorsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *MetricDescriptorsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/metricDescriptors")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudmonitoring.metricDescriptors.list" call.
// Exactly one of *ListMetricDescriptorsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListMetricDescriptorsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *MetricDescriptorsListCall) Do(opts ...googleapi.CallOption) (*ListMetricDescriptorsResponse, error) {
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
	ret := &ListMetricDescriptorsResponse{
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
	//   "description": "List metric descriptors that match the query. If the query is not set, then all of the metric descriptors will be returned. Large responses will be paginated, use the nextPageToken returned in the response to request subsequent pages of results by setting the pageToken query parameter to the value of the nextPageToken.",
	//   "httpMethod": "GET",
	//   "id": "cloudmonitoring.metricDescriptors.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "count": {
	//       "default": "100",
	//       "description": "Maximum number of metric descriptors per page. Used for pagination. If not specified, count = 100.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The pagination token, which is used to page through large result sets. Set this value to the value of the nextPageToken to retrieve the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project id. The value can be the numeric project ID or string-based project name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "query": {
	//       "description": "The query used to search against existing metrics. Separate keywords with a space; the service joins all keywords with AND, meaning that all keywords must match for a metric to be returned. If this field is omitted, all metrics are returned. If an empty string is passed with this field, no metrics are returned.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/metricDescriptors",
	//   "request": {
	//     "$ref": "ListMetricDescriptorsRequest"
	//   },
	//   "response": {
	//     "$ref": "ListMetricDescriptorsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/monitoring"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *MetricDescriptorsListCall) Pages(ctx context.Context, f func(*ListMetricDescriptorsResponse) error) error {
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

// method id "cloudmonitoring.timeseries.list":

type TimeseriesListCall struct {
	s                     *Service
	project               string
	metric                string
	listtimeseriesrequest *ListTimeseriesRequest
	urlParams_            gensupport.URLParams
	ifNoneMatch_          string
	ctx_                  context.Context
	header_               http.Header
}

// List: List the data points of the time series that match the metric
// and labels values and that have data points in the interval. Large
// responses are paginated; use the nextPageToken returned in the
// response to request subsequent pages of results by setting the
// pageToken query parameter to the value of the nextPageToken.
func (r *TimeseriesService) List(project string, metric string, youngest string, listtimeseriesrequest *ListTimeseriesRequest) *TimeseriesListCall {
	c := &TimeseriesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.metric = metric
	c.urlParams_.Set("youngest", youngest)
	c.listtimeseriesrequest = listtimeseriesrequest
	return c
}

// Aggregator sets the optional parameter "aggregator": The aggregation
// function that will reduce the data points in each window to a single
// point. This parameter is only valid for non-cumulative metrics with a
// value type of INT64 or DOUBLE.
//
// Possible values:
//   "max"
//   "mean"
//   "min"
//   "sum"
func (c *TimeseriesListCall) Aggregator(aggregator string) *TimeseriesListCall {
	c.urlParams_.Set("aggregator", aggregator)
	return c
}

// Count sets the optional parameter "count": Maximum number of data
// points per page, which is used for pagination of results.
func (c *TimeseriesListCall) Count(count int64) *TimeseriesListCall {
	c.urlParams_.Set("count", fmt.Sprint(count))
	return c
}

// Labels sets the optional parameter "labels": A collection of labels
// for the matching time series, which are represented as:
// - key==value: key equals the value
// - key=~value: key regex matches the value
// - key!=value: key does not equal the value
// - key!~value: key regex does not match the value  For example, to
// list all of the time series descriptors for the region us-central1,
// you could
// specify:
// label=cloud.googleapis.com%2Flocation=~us-central1.*
func (c *TimeseriesListCall) Labels(labels ...string) *TimeseriesListCall {
	c.urlParams_.SetMulti("labels", append([]string{}, labels...))
	return c
}

// Oldest sets the optional parameter "oldest": Start of the time
// interval (exclusive), which is expressed as an RFC 3339 timestamp. If
// neither oldest nor timespan is specified, the default time interval
// will be (youngest - 4 hours, youngest]
func (c *TimeseriesListCall) Oldest(oldest string) *TimeseriesListCall {
	c.urlParams_.Set("oldest", oldest)
	return c
}

// PageToken sets the optional parameter "pageToken": The pagination
// token, which is used to page through large result sets. Set this
// value to the value of the nextPageToken to retrieve the next page of
// results.
func (c *TimeseriesListCall) PageToken(pageToken string) *TimeseriesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Timespan sets the optional parameter "timespan": Length of the time
// interval to query, which is an alternative way to declare the
// interval: (youngest - timespan, youngest]. The timespan and oldest
// parameters should not be used together. Units:
// - s: second
// - m: minute
// - h: hour
// - d: day
// - w: week  Examples: 2s, 3m, 4w. Only one unit is allowed, for
// example: 2w3d is not allowed; you should use 17d instead.
//
// If neither oldest nor timespan is specified, the default time
// interval will be (youngest - 4 hours, youngest].
func (c *TimeseriesListCall) Timespan(timespan string) *TimeseriesListCall {
	c.urlParams_.Set("timespan", timespan)
	return c
}

// Window sets the optional parameter "window": The sampling window. At
// most one data point will be returned for each window in the requested
// time interval. This parameter is only valid for non-cumulative metric
// types. Units:
// - m: minute
// - h: hour
// - d: day
// - w: week  Examples: 3m, 4w. Only one unit is allowed, for example:
// 2w3d is not allowed; you should use 17d instead.
func (c *TimeseriesListCall) Window(window string) *TimeseriesListCall {
	c.urlParams_.Set("window", window)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TimeseriesListCall) Fields(s ...googleapi.Field) *TimeseriesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TimeseriesListCall) IfNoneMatch(entityTag string) *TimeseriesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TimeseriesListCall) Context(ctx context.Context) *TimeseriesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TimeseriesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TimeseriesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/timeseries/{metric}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"metric":  c.metric,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudmonitoring.timeseries.list" call.
// Exactly one of *ListTimeseriesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListTimeseriesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TimeseriesListCall) Do(opts ...googleapi.CallOption) (*ListTimeseriesResponse, error) {
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
	ret := &ListTimeseriesResponse{
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
	//   "description": "List the data points of the time series that match the metric and labels values and that have data points in the interval. Large responses are paginated; use the nextPageToken returned in the response to request subsequent pages of results by setting the pageToken query parameter to the value of the nextPageToken.",
	//   "httpMethod": "GET",
	//   "id": "cloudmonitoring.timeseries.list",
	//   "parameterOrder": [
	//     "project",
	//     "metric",
	//     "youngest"
	//   ],
	//   "parameters": {
	//     "aggregator": {
	//       "description": "The aggregation function that will reduce the data points in each window to a single point. This parameter is only valid for non-cumulative metrics with a value type of INT64 or DOUBLE.",
	//       "enum": [
	//         "max",
	//         "mean",
	//         "min",
	//         "sum"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "count": {
	//       "default": "6000",
	//       "description": "Maximum number of data points per page, which is used for pagination of results.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "12000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "labels": {
	//       "description": "A collection of labels for the matching time series, which are represented as:  \n- key==value: key equals the value \n- key=~value: key regex matches the value \n- key!=value: key does not equal the value \n- key!~value: key regex does not match the value  For example, to list all of the time series descriptors for the region us-central1, you could specify:\nlabel=cloud.googleapis.com%2Flocation=~us-central1.*",
	//       "location": "query",
	//       "pattern": "(.+?)(==|=~|!=|!~)(.+)",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "metric": {
	//       "description": "Metric names are protocol-free URLs as listed in the Supported Metrics page. For example, compute.googleapis.com/instance/disk/read_ops_count.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "oldest": {
	//       "description": "Start of the time interval (exclusive), which is expressed as an RFC 3339 timestamp. If neither oldest nor timespan is specified, the default time interval will be (youngest - 4 hours, youngest]",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pagination token, which is used to page through large result sets. Set this value to the value of the nextPageToken to retrieve the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID to which this time series belongs. The value can be the numeric project ID or string-based project name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "timespan": {
	//       "description": "Length of the time interval to query, which is an alternative way to declare the interval: (youngest - timespan, youngest]. The timespan and oldest parameters should not be used together. Units:  \n- s: second \n- m: minute \n- h: hour \n- d: day \n- w: week  Examples: 2s, 3m, 4w. Only one unit is allowed, for example: 2w3d is not allowed; you should use 17d instead.\n\nIf neither oldest nor timespan is specified, the default time interval will be (youngest - 4 hours, youngest].",
	//       "location": "query",
	//       "pattern": "[0-9]+[smhdw]?",
	//       "type": "string"
	//     },
	//     "window": {
	//       "description": "The sampling window. At most one data point will be returned for each window in the requested time interval. This parameter is only valid for non-cumulative metric types. Units:  \n- m: minute \n- h: hour \n- d: day \n- w: week  Examples: 3m, 4w. Only one unit is allowed, for example: 2w3d is not allowed; you should use 17d instead.",
	//       "location": "query",
	//       "pattern": "[0-9]+[mhdw]?",
	//       "type": "string"
	//     },
	//     "youngest": {
	//       "description": "End of the time interval (inclusive), which is expressed as an RFC 3339 timestamp.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/timeseries/{metric}",
	//   "request": {
	//     "$ref": "ListTimeseriesRequest"
	//   },
	//   "response": {
	//     "$ref": "ListTimeseriesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/monitoring"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TimeseriesListCall) Pages(ctx context.Context, f func(*ListTimeseriesResponse) error) error {
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

// method id "cloudmonitoring.timeseries.write":

type TimeseriesWriteCall struct {
	s                      *Service
	project                string
	writetimeseriesrequest *WriteTimeseriesRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Write: Put data points to one or more time series for one or more
// metrics. If a time series does not exist, a new time series will be
// created. It is not allowed to write a time series point that is older
// than the existing youngest point of that time series. Points that are
// older than the existing youngest point of that time series will be
// discarded silently. Therefore, users should make sure that points of
// a time series are written sequentially in the order of their end
// time.
func (r *TimeseriesService) Write(project string, writetimeseriesrequest *WriteTimeseriesRequest) *TimeseriesWriteCall {
	c := &TimeseriesWriteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.writetimeseriesrequest = writetimeseriesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TimeseriesWriteCall) Fields(s ...googleapi.Field) *TimeseriesWriteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TimeseriesWriteCall) Context(ctx context.Context) *TimeseriesWriteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TimeseriesWriteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TimeseriesWriteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.writetimeseriesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/timeseries:write")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudmonitoring.timeseries.write" call.
// Exactly one of *WriteTimeseriesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *WriteTimeseriesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TimeseriesWriteCall) Do(opts ...googleapi.CallOption) (*WriteTimeseriesResponse, error) {
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
	ret := &WriteTimeseriesResponse{
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
	//   "description": "Put data points to one or more time series for one or more metrics. If a time series does not exist, a new time series will be created. It is not allowed to write a time series point that is older than the existing youngest point of that time series. Points that are older than the existing youngest point of that time series will be discarded silently. Therefore, users should make sure that points of a time series are written sequentially in the order of their end time.",
	//   "httpMethod": "POST",
	//   "id": "cloudmonitoring.timeseries.write",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project ID. The value can be the numeric project ID or string-based project name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/timeseries:write",
	//   "request": {
	//     "$ref": "WriteTimeseriesRequest"
	//   },
	//   "response": {
	//     "$ref": "WriteTimeseriesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/monitoring"
	//   ]
	// }

}

// method id "cloudmonitoring.timeseriesDescriptors.list":

type TimeseriesDescriptorsListCall struct {
	s                                *Service
	project                          string
	metric                           string
	listtimeseriesdescriptorsrequest *ListTimeseriesDescriptorsRequest
	urlParams_                       gensupport.URLParams
	ifNoneMatch_                     string
	ctx_                             context.Context
	header_                          http.Header
}

// List: List the descriptors of the time series that match the metric
// and labels values and that have data points in the interval. Large
// responses are paginated; use the nextPageToken returned in the
// response to request subsequent pages of results by setting the
// pageToken query parameter to the value of the nextPageToken.
func (r *TimeseriesDescriptorsService) List(project string, metric string, youngest string, listtimeseriesdescriptorsrequest *ListTimeseriesDescriptorsRequest) *TimeseriesDescriptorsListCall {
	c := &TimeseriesDescriptorsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.metric = metric
	c.urlParams_.Set("youngest", youngest)
	c.listtimeseriesdescriptorsrequest = listtimeseriesdescriptorsrequest
	return c
}

// Aggregator sets the optional parameter "aggregator": The aggregation
// function that will reduce the data points in each window to a single
// point. This parameter is only valid for non-cumulative metrics with a
// value type of INT64 or DOUBLE.
//
// Possible values:
//   "max"
//   "mean"
//   "min"
//   "sum"
func (c *TimeseriesDescriptorsListCall) Aggregator(aggregator string) *TimeseriesDescriptorsListCall {
	c.urlParams_.Set("aggregator", aggregator)
	return c
}

// Count sets the optional parameter "count": Maximum number of time
// series descriptors per page. Used for pagination. If not specified,
// count = 100.
func (c *TimeseriesDescriptorsListCall) Count(count int64) *TimeseriesDescriptorsListCall {
	c.urlParams_.Set("count", fmt.Sprint(count))
	return c
}

// Labels sets the optional parameter "labels": A collection of labels
// for the matching time series, which are represented as:
// - key==value: key equals the value
// - key=~value: key regex matches the value
// - key!=value: key does not equal the value
// - key!~value: key regex does not match the value  For example, to
// list all of the time series descriptors for the region us-central1,
// you could
// specify:
// label=cloud.googleapis.com%2Flocation=~us-central1.*
func (c *TimeseriesDescriptorsListCall) Labels(labels ...string) *TimeseriesDescriptorsListCall {
	c.urlParams_.SetMulti("labels", append([]string{}, labels...))
	return c
}

// Oldest sets the optional parameter "oldest": Start of the time
// interval (exclusive), which is expressed as an RFC 3339 timestamp. If
// neither oldest nor timespan is specified, the default time interval
// will be (youngest - 4 hours, youngest]
func (c *TimeseriesDescriptorsListCall) Oldest(oldest string) *TimeseriesDescriptorsListCall {
	c.urlParams_.Set("oldest", oldest)
	return c
}

// PageToken sets the optional parameter "pageToken": The pagination
// token, which is used to page through large result sets. Set this
// value to the value of the nextPageToken to retrieve the next page of
// results.
func (c *TimeseriesDescriptorsListCall) PageToken(pageToken string) *TimeseriesDescriptorsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Timespan sets the optional parameter "timespan": Length of the time
// interval to query, which is an alternative way to declare the
// interval: (youngest - timespan, youngest]. The timespan and oldest
// parameters should not be used together. Units:
// - s: second
// - m: minute
// - h: hour
// - d: day
// - w: week  Examples: 2s, 3m, 4w. Only one unit is allowed, for
// example: 2w3d is not allowed; you should use 17d instead.
//
// If neither oldest nor timespan is specified, the default time
// interval will be (youngest - 4 hours, youngest].
func (c *TimeseriesDescriptorsListCall) Timespan(timespan string) *TimeseriesDescriptorsListCall {
	c.urlParams_.Set("timespan", timespan)
	return c
}

// Window sets the optional parameter "window": The sampling window. At
// most one data point will be returned for each window in the requested
// time interval. This parameter is only valid for non-cumulative metric
// types. Units:
// - m: minute
// - h: hour
// - d: day
// - w: week  Examples: 3m, 4w. Only one unit is allowed, for example:
// 2w3d is not allowed; you should use 17d instead.
func (c *TimeseriesDescriptorsListCall) Window(window string) *TimeseriesDescriptorsListCall {
	c.urlParams_.Set("window", window)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TimeseriesDescriptorsListCall) Fields(s ...googleapi.Field) *TimeseriesDescriptorsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TimeseriesDescriptorsListCall) IfNoneMatch(entityTag string) *TimeseriesDescriptorsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TimeseriesDescriptorsListCall) Context(ctx context.Context) *TimeseriesDescriptorsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TimeseriesDescriptorsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TimeseriesDescriptorsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/timeseriesDescriptors/{metric}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"metric":  c.metric,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudmonitoring.timeseriesDescriptors.list" call.
// Exactly one of *ListTimeseriesDescriptorsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListTimeseriesDescriptorsResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *TimeseriesDescriptorsListCall) Do(opts ...googleapi.CallOption) (*ListTimeseriesDescriptorsResponse, error) {
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
	ret := &ListTimeseriesDescriptorsResponse{
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
	//   "description": "List the descriptors of the time series that match the metric and labels values and that have data points in the interval. Large responses are paginated; use the nextPageToken returned in the response to request subsequent pages of results by setting the pageToken query parameter to the value of the nextPageToken.",
	//   "httpMethod": "GET",
	//   "id": "cloudmonitoring.timeseriesDescriptors.list",
	//   "parameterOrder": [
	//     "project",
	//     "metric",
	//     "youngest"
	//   ],
	//   "parameters": {
	//     "aggregator": {
	//       "description": "The aggregation function that will reduce the data points in each window to a single point. This parameter is only valid for non-cumulative metrics with a value type of INT64 or DOUBLE.",
	//       "enum": [
	//         "max",
	//         "mean",
	//         "min",
	//         "sum"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "count": {
	//       "default": "100",
	//       "description": "Maximum number of time series descriptors per page. Used for pagination. If not specified, count = 100.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "labels": {
	//       "description": "A collection of labels for the matching time series, which are represented as:  \n- key==value: key equals the value \n- key=~value: key regex matches the value \n- key!=value: key does not equal the value \n- key!~value: key regex does not match the value  For example, to list all of the time series descriptors for the region us-central1, you could specify:\nlabel=cloud.googleapis.com%2Flocation=~us-central1.*",
	//       "location": "query",
	//       "pattern": "(.+?)(==|=~|!=|!~)(.+)",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "metric": {
	//       "description": "Metric names are protocol-free URLs as listed in the Supported Metrics page. For example, compute.googleapis.com/instance/disk/read_ops_count.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "oldest": {
	//       "description": "Start of the time interval (exclusive), which is expressed as an RFC 3339 timestamp. If neither oldest nor timespan is specified, the default time interval will be (youngest - 4 hours, youngest]",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pagination token, which is used to page through large result sets. Set this value to the value of the nextPageToken to retrieve the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID to which this time series belongs. The value can be the numeric project ID or string-based project name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "timespan": {
	//       "description": "Length of the time interval to query, which is an alternative way to declare the interval: (youngest - timespan, youngest]. The timespan and oldest parameters should not be used together. Units:  \n- s: second \n- m: minute \n- h: hour \n- d: day \n- w: week  Examples: 2s, 3m, 4w. Only one unit is allowed, for example: 2w3d is not allowed; you should use 17d instead.\n\nIf neither oldest nor timespan is specified, the default time interval will be (youngest - 4 hours, youngest].",
	//       "location": "query",
	//       "pattern": "[0-9]+[smhdw]?",
	//       "type": "string"
	//     },
	//     "window": {
	//       "description": "The sampling window. At most one data point will be returned for each window in the requested time interval. This parameter is only valid for non-cumulative metric types. Units:  \n- m: minute \n- h: hour \n- d: day \n- w: week  Examples: 3m, 4w. Only one unit is allowed, for example: 2w3d is not allowed; you should use 17d instead.",
	//       "location": "query",
	//       "pattern": "[0-9]+[mhdw]?",
	//       "type": "string"
	//     },
	//     "youngest": {
	//       "description": "End of the time interval (inclusive), which is expressed as an RFC 3339 timestamp.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/timeseriesDescriptors/{metric}",
	//   "request": {
	//     "$ref": "ListTimeseriesDescriptorsRequest"
	//   },
	//   "response": {
	//     "$ref": "ListTimeseriesDescriptorsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/monitoring"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TimeseriesDescriptorsListCall) Pages(ctx context.Context, f func(*ListTimeseriesDescriptorsResponse) error) error {
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
