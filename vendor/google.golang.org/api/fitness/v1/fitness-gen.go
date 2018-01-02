// Package fitness provides access to the Fitness.
//
// See https://developers.google.com/fit/rest/
//
// Usage example:
//
//   import "google.golang.org/api/fitness/v1"
//   ...
//   fitnessService, err := fitness.New(oauthHttpClient)
package fitness // import "google.golang.org/api/fitness/v1"

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

const apiId = "fitness:v1"
const apiName = "fitness"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/fitness/v1/users/"

// OAuth2 scopes used by this API.
const (
	// View your activity information in Google Fit
	FitnessActivityReadScope = "https://www.googleapis.com/auth/fitness.activity.read"

	// View and store your activity information in Google Fit
	FitnessActivityWriteScope = "https://www.googleapis.com/auth/fitness.activity.write"

	// View blood glucose data in Google Fit
	FitnessBloodGlucoseReadScope = "https://www.googleapis.com/auth/fitness.blood_glucose.read"

	// View and store blood glucose data in Google Fit
	FitnessBloodGlucoseWriteScope = "https://www.googleapis.com/auth/fitness.blood_glucose.write"

	// View blood pressure data in Google Fit
	FitnessBloodPressureReadScope = "https://www.googleapis.com/auth/fitness.blood_pressure.read"

	// View and store blood pressure data in Google Fit
	FitnessBloodPressureWriteScope = "https://www.googleapis.com/auth/fitness.blood_pressure.write"

	// View body sensor information in Google Fit
	FitnessBodyReadScope = "https://www.googleapis.com/auth/fitness.body.read"

	// View and store body sensor data in Google Fit
	FitnessBodyWriteScope = "https://www.googleapis.com/auth/fitness.body.write"

	// View body temperature data in Google Fit
	FitnessBodyTemperatureReadScope = "https://www.googleapis.com/auth/fitness.body_temperature.read"

	// View and store body temperature data in Google Fit
	FitnessBodyTemperatureWriteScope = "https://www.googleapis.com/auth/fitness.body_temperature.write"

	// View your stored location data in Google Fit
	FitnessLocationReadScope = "https://www.googleapis.com/auth/fitness.location.read"

	// View and store your location data in Google Fit
	FitnessLocationWriteScope = "https://www.googleapis.com/auth/fitness.location.write"

	// View nutrition information in Google Fit
	FitnessNutritionReadScope = "https://www.googleapis.com/auth/fitness.nutrition.read"

	// View and store nutrition information in Google Fit
	FitnessNutritionWriteScope = "https://www.googleapis.com/auth/fitness.nutrition.write"

	// View oxygen saturation data in Google Fit
	FitnessOxygenSaturationReadScope = "https://www.googleapis.com/auth/fitness.oxygen_saturation.read"

	// View and store oxygen saturation data in Google Fit
	FitnessOxygenSaturationWriteScope = "https://www.googleapis.com/auth/fitness.oxygen_saturation.write"

	// View reproductive health data in Google Fit
	FitnessReproductiveHealthReadScope = "https://www.googleapis.com/auth/fitness.reproductive_health.read"

	// View and store reproductive health data in Google Fit
	FitnessReproductiveHealthWriteScope = "https://www.googleapis.com/auth/fitness.reproductive_health.write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Users *UsersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewUsersService(s *Service) *UsersService {
	rs := &UsersService{s: s}
	rs.DataSources = NewUsersDataSourcesService(s)
	rs.Dataset = NewUsersDatasetService(s)
	rs.Sessions = NewUsersSessionsService(s)
	return rs
}

type UsersService struct {
	s *Service

	DataSources *UsersDataSourcesService

	Dataset *UsersDatasetService

	Sessions *UsersSessionsService
}

func NewUsersDataSourcesService(s *Service) *UsersDataSourcesService {
	rs := &UsersDataSourcesService{s: s}
	rs.Datasets = NewUsersDataSourcesDatasetsService(s)
	return rs
}

type UsersDataSourcesService struct {
	s *Service

	Datasets *UsersDataSourcesDatasetsService
}

func NewUsersDataSourcesDatasetsService(s *Service) *UsersDataSourcesDatasetsService {
	rs := &UsersDataSourcesDatasetsService{s: s}
	return rs
}

type UsersDataSourcesDatasetsService struct {
	s *Service
}

func NewUsersDatasetService(s *Service) *UsersDatasetService {
	rs := &UsersDatasetService{s: s}
	return rs
}

type UsersDatasetService struct {
	s *Service
}

func NewUsersSessionsService(s *Service) *UsersSessionsService {
	rs := &UsersSessionsService{s: s}
	return rs
}

type UsersSessionsService struct {
	s *Service
}

type AggregateBucket struct {
	// Activity: Available for Bucket.Type.ACTIVITY_TYPE,
	// Bucket.Type.ACTIVITY_SEGMENT
	Activity int64 `json:"activity,omitempty"`

	// Dataset: There will be one dataset per AggregateBy in the request.
	Dataset []*Dataset `json:"dataset,omitempty"`

	// EndTimeMillis: The end time for the aggregated data, in milliseconds
	// since epoch, inclusive.
	EndTimeMillis int64 `json:"endTimeMillis,omitempty,string"`

	// Session: Available for Bucket.Type.SESSION
	Session *Session `json:"session,omitempty"`

	// StartTimeMillis: The start time for the aggregated data, in
	// milliseconds since epoch, inclusive.
	StartTimeMillis int64 `json:"startTimeMillis,omitempty,string"`

	// Type: The type of a bucket signifies how the data aggregation is
	// performed in the bucket.
	//
	// Possible values:
	//   "activitySegment"
	//   "activityType"
	//   "session"
	//   "time"
	//   "unknown"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Activity") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Activity") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AggregateBucket) MarshalJSON() ([]byte, error) {
	type noMethod AggregateBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AggregateBy: The specification of which data to aggregate.
type AggregateBy struct {
	// DataSourceId: A data source ID to aggregate. Mutually exclusive of
	// dataTypeName. Only data from the specified data source ID will be
	// included in the aggregation. The dataset in the response will have
	// the same data source ID.
	DataSourceId string `json:"dataSourceId,omitempty"`

	// DataTypeName: The data type to aggregate. All data sources providing
	// this data type will contribute data to the aggregation. The response
	// will contain a single dataset for this data type name. The dataset
	// will have a data source ID of
	// derived:com.google.:com.google.android.gms:aggregated
	DataTypeName string `json:"dataTypeName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DataSourceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DataSourceId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AggregateBy) MarshalJSON() ([]byte, error) {
	type noMethod AggregateBy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AggregateRequest: Next id: 10
type AggregateRequest struct {
	// AggregateBy: The specification of data to be aggregated. At least one
	// aggregateBy spec must be provided. All data that is specified will be
	// aggregated using the same bucketing criteria. There will be one
	// dataset in the response for every aggregateBy spec.
	AggregateBy []*AggregateBy `json:"aggregateBy,omitempty"`

	// BucketByActivitySegment: Specifies that data be aggregated each
	// activity segment recored for a user. Similar to
	// bucketByActivitySegment, but bucketing is done for each activity
	// segment rather than all segments of the same type. Mutually exclusive
	// of other bucketing specifications.
	BucketByActivitySegment *BucketByActivity `json:"bucketByActivitySegment,omitempty"`

	// BucketByActivityType: Specifies that data be aggregated by the type
	// of activity being performed when the data was recorded. All data that
	// was recorded during a certain activity type (for the given time
	// range) will be aggregated into the same bucket. Data that was
	// recorded while the user was not active will not be included in the
	// response. Mutually exclusive of other bucketing specifications.
	BucketByActivityType *BucketByActivity `json:"bucketByActivityType,omitempty"`

	// BucketBySession: Specifies that data be aggregated by user sessions.
	// Data that does not fall within the time range of a session will not
	// be included in the response. Mutually exclusive of other bucketing
	// specifications.
	BucketBySession *BucketBySession `json:"bucketBySession,omitempty"`

	// BucketByTime: Specifies that data be aggregated by a single time
	// interval. Mutually exclusive of other bucketing specifications.
	BucketByTime *BucketByTime `json:"bucketByTime,omitempty"`

	// EndTimeMillis: The end of a window of time. Data that intersects with
	// this time window will be aggregated. The time is in milliseconds
	// since epoch, inclusive.
	EndTimeMillis int64 `json:"endTimeMillis,omitempty,string"`

	// FilteredDataQualityStandard: A list of acceptable data quality
	// standards. Only data points which conform to at least one of the
	// specified data quality standards will be returned. If the list is
	// empty, all data points are returned.
	//
	// Possible values:
	//   "dataQualityBloodGlucoseIso151972003"
	//   "dataQualityBloodGlucoseIso151972013"
	//   "dataQualityBloodPressureAami"
	//   "dataQualityBloodPressureBhsAA"
	//   "dataQualityBloodPressureBhsAB"
	//   "dataQualityBloodPressureBhsBA"
	//   "dataQualityBloodPressureBhsBB"
	//   "dataQualityBloodPressureEsh2002"
	//   "dataQualityBloodPressureEsh2010"
	//   "dataQualityUnknown"
	FilteredDataQualityStandard []string `json:"filteredDataQualityStandard,omitempty"`

	// StartTimeMillis: The start of a window of time. Data that intersects
	// with this time window will be aggregated. The time is in milliseconds
	// since epoch, inclusive.
	StartTimeMillis int64 `json:"startTimeMillis,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "AggregateBy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AggregateBy") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AggregateRequest) MarshalJSON() ([]byte, error) {
	type noMethod AggregateRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type AggregateResponse struct {
	// Bucket: A list of buckets containing the aggregated data.
	Bucket []*AggregateBucket `json:"bucket,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Bucket") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bucket") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AggregateResponse) MarshalJSON() ([]byte, error) {
	type noMethod AggregateResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Application struct {
	// DetailsUrl: An optional URI that can be used to link back to the
	// application.
	DetailsUrl string `json:"detailsUrl,omitempty"`

	// Name: The name of this application. This is required for REST
	// clients, but we do not enforce uniqueness of this name. It is
	// provided as a matter of convenience for other developers who would
	// like to identify which REST created an Application or Data Source.
	Name string `json:"name,omitempty"`

	// PackageName: Package name for this application. This is used as a
	// unique identifier when created by Android applications, but cannot be
	// specified by REST clients. REST clients will have their developer
	// project number reflected into the Data Source data stream IDs,
	// instead of the packageName.
	PackageName string `json:"packageName,omitempty"`

	// Version: Version of the application. You should update this field
	// whenever the application changes in a way that affects the
	// computation of the data.
	Version string `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DetailsUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DetailsUrl") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Application) MarshalJSON() ([]byte, error) {
	type noMethod Application
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BucketByActivity struct {
	// ActivityDataSourceId: The default activity stream will be used if a
	// specific activityDataSourceId is not specified.
	ActivityDataSourceId string `json:"activityDataSourceId,omitempty"`

	// MinDurationMillis: Specifies that only activity segments of duration
	// longer than minDurationMillis are considered and used as a container
	// for aggregated data.
	MinDurationMillis int64 `json:"minDurationMillis,omitempty,string"`

	// ForceSendFields is a list of field names (e.g.
	// "ActivityDataSourceId") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ActivityDataSourceId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BucketByActivity) MarshalJSON() ([]byte, error) {
	type noMethod BucketByActivity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BucketBySession struct {
	// MinDurationMillis: Specifies that only sessions of duration longer
	// than minDurationMillis are considered and used as a container for
	// aggregated data.
	MinDurationMillis int64 `json:"minDurationMillis,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "MinDurationMillis")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MinDurationMillis") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BucketBySession) MarshalJSON() ([]byte, error) {
	type noMethod BucketBySession
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BucketByTime struct {
	// DurationMillis: Specifies that result buckets aggregate data by
	// exactly durationMillis time frames. Time frames that contain no data
	// will be included in the response with an empty dataset.
	DurationMillis int64 `json:"durationMillis,omitempty,string"`

	Period *BucketByTimePeriod `json:"period,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DurationMillis") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DurationMillis") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BucketByTime) MarshalJSON() ([]byte, error) {
	type noMethod BucketByTime
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BucketByTimePeriod struct {
	// TimeZoneId: org.joda.timezone.DateTimeZone
	TimeZoneId string `json:"timeZoneId,omitempty"`

	// Possible values:
	//   "day"
	//   "month"
	//   "week"
	Type string `json:"type,omitempty"`

	Value int64 `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TimeZoneId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TimeZoneId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BucketByTimePeriod) MarshalJSON() ([]byte, error) {
	type noMethod BucketByTimePeriod
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DataPoint: Represents a single data point, generated by a particular
// data source. A data point holds a value for each field, an end
// timestamp and an optional start time. The exact semantics of each of
// these attributes are specified in the documentation for the
// particular data type.
//
// A data point can represent an instantaneous measurement, reading or
// input observation, as well as averages or aggregates over a time
// interval. Check the data type documentation to determine which is the
// case for a particular data type.
//
// Data points always contain one value for each field of the data type.
type DataPoint struct {
	// ComputationTimeMillis: Used for version checking during
	// transformation; that is, a datapoint can only replace another
	// datapoint that has an older computation time stamp.
	ComputationTimeMillis int64 `json:"computationTimeMillis,omitempty,string"`

	// DataTypeName: The data type defining the format of the values in this
	// data point.
	DataTypeName string `json:"dataTypeName,omitempty"`

	// EndTimeNanos: The end time of the interval represented by this data
	// point, in nanoseconds since epoch.
	EndTimeNanos int64 `json:"endTimeNanos,omitempty,string"`

	// ModifiedTimeMillis: Indicates the last time this data point was
	// modified. Useful only in contexts where we are listing the data
	// changes, rather than representing the current state of the data.
	ModifiedTimeMillis int64 `json:"modifiedTimeMillis,omitempty,string"`

	// OriginDataSourceId: If the data point is contained in a dataset for a
	// derived data source, this field will be populated with the data
	// source stream ID that created the data point originally.
	OriginDataSourceId string `json:"originDataSourceId,omitempty"`

	// RawTimestampNanos: The raw timestamp from the original SensorEvent.
	RawTimestampNanos int64 `json:"rawTimestampNanos,omitempty,string"`

	// StartTimeNanos: The start time of the interval represented by this
	// data point, in nanoseconds since epoch.
	StartTimeNanos int64 `json:"startTimeNanos,omitempty,string"`

	// Value: Values of each data type field for the data point. It is
	// expected that each value corresponding to a data type field will
	// occur in the same order that the field is listed with in the data
	// type specified in a data source.
	//
	// Only one of integer and floating point fields will be populated,
	// depending on the format enum value within data source's type field.
	Value []*Value `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ComputationTimeMillis") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ComputationTimeMillis") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DataPoint) MarshalJSON() ([]byte, error) {
	type noMethod DataPoint
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DataSource: Definition of a unique source of sensor data. Data
// sources can expose raw data coming from hardware sensors on local or
// companion devices. They can also expose derived data, created by
// transforming or merging other data sources. Multiple data sources can
// exist for the same data type. Every data point inserted into or read
// from this service has an associated data source.
//
// The data source contains enough information to uniquely identify its
// data, including the hardware device and the application that
// collected and/or transformed the data. It also holds useful metadata,
// such as the hardware and application versions, and the device
// type.
//
// Each data source produces a unique stream of data, with a unique
// identifier. Not all changes to data source affect the stream
// identifier, so that data collected by updated versions of the same
// application/device can still be considered to belong to the same data
// stream.
type DataSource struct {
	// Application: Information about an application which feeds sensor data
	// into the platform.
	Application *Application `json:"application,omitempty"`

	// Possible values:
	//   "dataQualityBloodGlucoseIso151972003"
	//   "dataQualityBloodGlucoseIso151972013"
	//   "dataQualityBloodPressureAami"
	//   "dataQualityBloodPressureBhsAA"
	//   "dataQualityBloodPressureBhsAB"
	//   "dataQualityBloodPressureBhsBA"
	//   "dataQualityBloodPressureBhsBB"
	//   "dataQualityBloodPressureEsh2002"
	//   "dataQualityBloodPressureEsh2010"
	//   "dataQualityUnknown"
	DataQualityStandard []string `json:"dataQualityStandard,omitempty"`

	// DataStreamId: A unique identifier for the data stream produced by
	// this data source. The identifier includes:
	//
	//
	// - The physical device's manufacturer, model, and serial number (UID).
	//
	// - The application's package name or name. Package name is used when
	// the data source was created by an Android application. The developer
	// project number is used when the data source was created by a REST
	// client.
	// - The data source's type.
	// - The data source's stream name.  Note that not all attributes of the
	// data source are used as part of the stream identifier. In particular,
	// the version of the hardware/the application isn't used. This allows
	// us to preserve the same stream through version updates. This also
	// means that two DataSource objects may represent the same data stream
	// even if they're not equal.
	//
	// The exact format of the data stream ID created by an Android
	// application is:
	// type:dataType.name:application.packageName:device.manufacturer:device.
	// model:device.uid:dataStreamName
	//
	// The exact format of the data stream ID created by a REST client is:
	// type:dataType.name:developer project
	// number:device.manufacturer:device.model:device.uid:dataStreamName
	//
	//
	// When any of the optional fields that comprise of the data stream ID
	// are blank, they will be omitted from the data stream ID. The minimum
	// viable data stream ID would be: type:dataType.name:developer project
	// number
	//
	// Finally, the developer project number is obfuscated when read by any
	// REST or Android client that did not create the data source. Only the
	// data source creator will see the developer project number in clear
	// and normal form.
	DataStreamId string `json:"dataStreamId,omitempty"`

	// DataStreamName: The stream name uniquely identifies this particular
	// data source among other data sources of the same type from the same
	// underlying producer. Setting the stream name is optional, but should
	// be done whenever an application exposes two streams for the same data
	// type, or when a device has two equivalent sensors.
	DataStreamName string `json:"dataStreamName,omitempty"`

	// DataType: The data type defines the schema for a stream of data being
	// collected by, inserted into, or queried from the Fitness API.
	DataType *DataType `json:"dataType,omitempty"`

	// Device: Representation of an integrated device (such as a phone or a
	// wearable) that can hold sensors.
	Device *Device `json:"device,omitempty"`

	// Name: An end-user visible name for this data source.
	Name string `json:"name,omitempty"`

	// Type: A constant describing the type of this data source. Indicates
	// whether this data source produces raw or derived data.
	//
	// Possible values:
	//   "derived"
	//   "raw"
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Application") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Application") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DataSource) MarshalJSON() ([]byte, error) {
	type noMethod DataSource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DataType struct {
	// Field: A field represents one dimension of a data type.
	Field []*DataTypeField `json:"field,omitempty"`

	// Name: Each data type has a unique, namespaced, name. All data types
	// in the com.google namespace are shared as part of the platform.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Field") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Field") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DataType) MarshalJSON() ([]byte, error) {
	type noMethod DataType
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DataTypeField: In case of multi-dimensional data (such as an
// accelerometer with x, y, and z axes) each field represents one
// dimension. Each data type field has a unique name which identifies
// it. The field also defines the format of the data (int, float,
// etc.).
//
// This message is only instantiated in code and not used for wire comms
// or stored in any way.
type DataTypeField struct {
	// Format: The different supported formats for each field in a data
	// type.
	//
	// Possible values:
	//   "blob"
	//   "floatList"
	//   "floatPoint"
	//   "integer"
	//   "integerList"
	//   "map"
	//   "string"
	Format string `json:"format,omitempty"`

	// Name: Defines the name and format of data. Unlike data type names,
	// field names are not namespaced, and only need to be unique within the
	// data type.
	Name string `json:"name,omitempty"`

	Optional bool `json:"optional,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Format") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Format") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DataTypeField) MarshalJSON() ([]byte, error) {
	type noMethod DataTypeField
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Dataset: A dataset represents a projection container for data points.
// They do not carry any info of their own. Datasets represent a set of
// data points from a particular data source. A data point can be found
// in more than one dataset.
type Dataset struct {
	// DataSourceId: The data stream ID of the data source that created the
	// points in this dataset.
	DataSourceId string `json:"dataSourceId,omitempty"`

	// MaxEndTimeNs: The largest end time of all data points in this
	// possibly partial representation of the dataset. Time is in
	// nanoseconds from epoch. This should also match the first part of the
	// dataset identifier.
	MaxEndTimeNs int64 `json:"maxEndTimeNs,omitempty,string"`

	// MinStartTimeNs: The smallest start time of all data points in this
	// possibly partial representation of the dataset. Time is in
	// nanoseconds from epoch. This should also match the first part of the
	// dataset identifier.
	MinStartTimeNs int64 `json:"minStartTimeNs,omitempty,string"`

	// NextPageToken: This token will be set when a dataset is received in
	// response to a GET request and the dataset is too large to be included
	// in a single response. Provide this value in a subsequent GET request
	// to return the next page of data points within this dataset.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Point: A partial list of data points contained in the dataset,
	// ordered by largest endTimeNanos first. This list is considered
	// complete when retrieving a small dataset and partial when patching a
	// dataset or retrieving a dataset that is too large to include in a
	// single response.
	Point []*DataPoint `json:"point,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DataSourceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DataSourceId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Dataset) MarshalJSON() ([]byte, error) {
	type noMethod Dataset
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Device: Representation of an integrated device (such as a phone or a
// wearable) that can hold sensors. Each sensor is exposed as a data
// source.
//
// The main purpose of the device information contained in this class is
// to identify the hardware of a particular data source. This can be
// useful in different ways, including:
// - Distinguishing two similar sensors on different devices (the step
// counter on two nexus 5 phones, for instance)
// - Display the source of data to the user (by using the device make /
// model)
// - Treat data differently depending on sensor type (accelerometers on
// a watch may give different patterns than those on a phone)
// - Build different analysis models for each device/version.
type Device struct {
	// Manufacturer: Manufacturer of the product/hardware.
	Manufacturer string `json:"manufacturer,omitempty"`

	// Model: End-user visible model name for the device.
	Model string `json:"model,omitempty"`

	// Type: A constant representing the type of the device.
	//
	// Possible values:
	//   "chestStrap"
	//   "headMounted"
	//   "phone"
	//   "scale"
	//   "tablet"
	//   "unknown"
	//   "watch"
	Type string `json:"type,omitempty"`

	// Uid: The serial number or other unique ID for the hardware. This
	// field is obfuscated when read by any REST or Android client that did
	// not create the data source. Only the data source creator will see the
	// uid field in clear and normal form.
	Uid string `json:"uid,omitempty"`

	// Version: Version string for the device hardware/software.
	Version string `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Manufacturer") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Manufacturer") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Device) MarshalJSON() ([]byte, error) {
	type noMethod Device
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListDataSourcesResponse struct {
	// DataSource: A previously created data source.
	DataSource []*DataSource `json:"dataSource,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DataSource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DataSource") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListDataSourcesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListDataSourcesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListSessionsResponse struct {
	// DeletedSession: If includeDeleted is set to true in the request, this
	// list will contain sessions deleted with original end times that are
	// within the startTime and endTime frame.
	DeletedSession []*Session `json:"deletedSession,omitempty"`

	// HasMoreData: Flag to indicate server has more data to transfer
	HasMoreData bool `json:"hasMoreData,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets. Provide this value in a subsequent request to
	// return the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Session: Sessions with an end time that is between startTime and
	// endTime of the request.
	Session []*Session `json:"session,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DeletedSession") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeletedSession") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListSessionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListSessionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MapValue: Holder object for the value of an entry in a map field of a
// data point.
//
// A map value supports a subset of the formats that the regular Value
// supports.
type MapValue struct {
	// FpVal: Floating point value.
	FpVal float64 `json:"fpVal,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FpVal") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FpVal") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MapValue) MarshalJSON() ([]byte, error) {
	type noMethod MapValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *MapValue) UnmarshalJSON(data []byte) error {
	type noMethod MapValue
	var s1 struct {
		FpVal gensupport.JSONFloat64 `json:"fpVal"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.FpVal = float64(s1.FpVal)
	return nil
}

// Session: Sessions contain metadata, such as a user-friendly name and
// time interval information.
type Session struct {
	// ActiveTimeMillis: Session active time. While start_time_millis and
	// end_time_millis define the full session time, the active time can be
	// shorter and specified by active_time_millis. If the inactive time
	// during the session is known, it should also be inserted via a
	// com.google.activity.segment data point with a STILL activity value
	ActiveTimeMillis int64 `json:"activeTimeMillis,omitempty,string"`

	// ActivityType: The type of activity this session represents.
	ActivityType int64 `json:"activityType,omitempty"`

	// Application: The application that created the session.
	Application *Application `json:"application,omitempty"`

	// Description: A description for this session.
	Description string `json:"description,omitempty"`

	// EndTimeMillis: An end time, in milliseconds since epoch, inclusive.
	EndTimeMillis int64 `json:"endTimeMillis,omitempty,string"`

	// Id: A client-generated identifier that is unique across all sessions
	// owned by this particular user.
	Id string `json:"id,omitempty"`

	// ModifiedTimeMillis: A timestamp that indicates when the session was
	// last modified.
	ModifiedTimeMillis int64 `json:"modifiedTimeMillis,omitempty,string"`

	// Name: A human readable name of the session.
	Name string `json:"name,omitempty"`

	// StartTimeMillis: A start time, in milliseconds since epoch,
	// inclusive.
	StartTimeMillis int64 `json:"startTimeMillis,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ActiveTimeMillis") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ActiveTimeMillis") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Session) MarshalJSON() ([]byte, error) {
	type noMethod Session
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Value: Holder object for the value of a single field in a data
// point.
//
// A field value has a particular format and is only ever set to one of
// an integer or a floating point value. LINT.IfChange
type Value struct {
	// FpVal: Floating point value. When this is set, other values must not
	// be set.
	FpVal float64 `json:"fpVal,omitempty"`

	// IntVal: Integer value. When this is set, other values must not be
	// set.
	IntVal int64 `json:"intVal,omitempty"`

	// MapVal: Map value. The valid key space and units for the
	// corresponding value of each entry should be documented as part of the
	// data type definition. Keys should be kept small whenever possible.
	// Data streams with large keys and high data frequency may be down
	// sampled.
	MapVal []*ValueMapValEntry `json:"mapVal,omitempty"`

	// StringVal: String value. When this is set, other values must not be
	// set. Strings should be kept small whenever possible. Data streams
	// with large string values and high data frequency may be down sampled.
	StringVal string `json:"stringVal,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FpVal") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FpVal") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Value) MarshalJSON() ([]byte, error) {
	type noMethod Value
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Value) UnmarshalJSON(data []byte) error {
	type noMethod Value
	var s1 struct {
		FpVal gensupport.JSONFloat64 `json:"fpVal"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.FpVal = float64(s1.FpVal)
	return nil
}

type ValueMapValEntry struct {
	Key string `json:"key,omitempty"`

	Value *MapValue `json:"value,omitempty"`

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

func (s *ValueMapValEntry) MarshalJSON() ([]byte, error) {
	type noMethod ValueMapValEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "fitness.users.dataSources.create":

type UsersDataSourcesCreateCall struct {
	s          *Service
	userId     string
	datasource *DataSource
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new data source that is unique across all data
// sources belonging to this user. The data stream ID field can be
// omitted and will be generated by the server with the correct format.
// The data stream ID is an ordered combination of some fields from the
// data source. In addition to the data source fields reflected into the
// data source ID, the developer project number that is authenticated
// when creating the data source is included. This developer project
// number is obfuscated when read by any other developer reading public
// data types.
func (r *UsersDataSourcesService) Create(userId string, datasource *DataSource) *UsersDataSourcesCreateCall {
	c := &UsersDataSourcesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.datasource = datasource
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesCreateCall) Fields(s ...googleapi.Field) *UsersDataSourcesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesCreateCall) Context(ctx context.Context) *UsersDataSourcesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datasource)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.create" call.
// Exactly one of *DataSource or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DataSource.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersDataSourcesCreateCall) Do(opts ...googleapi.CallOption) (*DataSource, error) {
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
	ret := &DataSource{
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
	//   "description": "Creates a new data source that is unique across all data sources belonging to this user. The data stream ID field can be omitted and will be generated by the server with the correct format. The data stream ID is an ordered combination of some fields from the data source. In addition to the data source fields reflected into the data source ID, the developer project number that is authenticated when creating the data source is included. This developer project number is obfuscated when read by any other developer reading public data types.",
	//   "httpMethod": "POST",
	//   "id": "fitness.users.dataSources.create",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "Create the data source for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources",
	//   "request": {
	//     "$ref": "DataSource"
	//   },
	//   "response": {
	//     "$ref": "DataSource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.dataSources.delete":

type UsersDataSourcesDeleteCall struct {
	s            *Service
	userId       string
	dataSourceId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deletes the specified data source. The request will fail if
// the data source contains any data points.
func (r *UsersDataSourcesService) Delete(userId string, dataSourceId string) *UsersDataSourcesDeleteCall {
	c := &UsersDataSourcesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.dataSourceId = dataSourceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesDeleteCall) Fields(s ...googleapi.Field) *UsersDataSourcesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesDeleteCall) Context(ctx context.Context) *UsersDataSourcesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources/{dataSourceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":       c.userId,
		"dataSourceId": c.dataSourceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.delete" call.
// Exactly one of *DataSource or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DataSource.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersDataSourcesDeleteCall) Do(opts ...googleapi.CallOption) (*DataSource, error) {
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
	ret := &DataSource{
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
	//   "description": "Deletes the specified data source. The request will fail if the data source contains any data points.",
	//   "httpMethod": "DELETE",
	//   "id": "fitness.users.dataSources.delete",
	//   "parameterOrder": [
	//     "userId",
	//     "dataSourceId"
	//   ],
	//   "parameters": {
	//     "dataSourceId": {
	//       "description": "The data stream ID of the data source to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Retrieve a data source for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources/{dataSourceId}",
	//   "response": {
	//     "$ref": "DataSource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.dataSources.get":

type UsersDataSourcesGetCall struct {
	s            *Service
	userId       string
	dataSourceId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns the specified data source.
func (r *UsersDataSourcesService) Get(userId string, dataSourceId string) *UsersDataSourcesGetCall {
	c := &UsersDataSourcesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.dataSourceId = dataSourceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesGetCall) Fields(s ...googleapi.Field) *UsersDataSourcesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersDataSourcesGetCall) IfNoneMatch(entityTag string) *UsersDataSourcesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesGetCall) Context(ctx context.Context) *UsersDataSourcesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources/{dataSourceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":       c.userId,
		"dataSourceId": c.dataSourceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.get" call.
// Exactly one of *DataSource or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DataSource.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersDataSourcesGetCall) Do(opts ...googleapi.CallOption) (*DataSource, error) {
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
	ret := &DataSource{
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
	//   "description": "Returns the specified data source.",
	//   "httpMethod": "GET",
	//   "id": "fitness.users.dataSources.get",
	//   "parameterOrder": [
	//     "userId",
	//     "dataSourceId"
	//   ],
	//   "parameters": {
	//     "dataSourceId": {
	//       "description": "The data stream ID of the data source to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Retrieve a data source for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources/{dataSourceId}",
	//   "response": {
	//     "$ref": "DataSource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.read",
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.read",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.read",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.read",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.read",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.read",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.read",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.read",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.dataSources.list":

type UsersDataSourcesListCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all data sources that are visible to the developer, using
// the OAuth scopes provided. The list is not exhaustive; the user may
// have private data sources that are only visible to other developers,
// or calls using other scopes.
func (r *UsersDataSourcesService) List(userId string) *UsersDataSourcesListCall {
	c := &UsersDataSourcesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// DataTypeName sets the optional parameter "dataTypeName": The names of
// data types to include in the list. If not specified, all data sources
// will be returned.
func (c *UsersDataSourcesListCall) DataTypeName(dataTypeName ...string) *UsersDataSourcesListCall {
	c.urlParams_.SetMulti("dataTypeName", append([]string{}, dataTypeName...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesListCall) Fields(s ...googleapi.Field) *UsersDataSourcesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersDataSourcesListCall) IfNoneMatch(entityTag string) *UsersDataSourcesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesListCall) Context(ctx context.Context) *UsersDataSourcesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.list" call.
// Exactly one of *ListDataSourcesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListDataSourcesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersDataSourcesListCall) Do(opts ...googleapi.CallOption) (*ListDataSourcesResponse, error) {
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
	ret := &ListDataSourcesResponse{
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
	//   "description": "Lists all data sources that are visible to the developer, using the OAuth scopes provided. The list is not exhaustive; the user may have private data sources that are only visible to other developers, or calls using other scopes.",
	//   "httpMethod": "GET",
	//   "id": "fitness.users.dataSources.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "dataTypeName": {
	//       "description": "The names of data types to include in the list. If not specified, all data sources will be returned.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "List data sources for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources",
	//   "response": {
	//     "$ref": "ListDataSourcesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.read",
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.read",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.read",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.read",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.read",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.read",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.read",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.read",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.dataSources.patch":

type UsersDataSourcesPatchCall struct {
	s            *Service
	userId       string
	dataSourceId string
	datasource   *DataSource
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Patch: Updates the specified data source. The dataStreamId, dataType,
// type, dataStreamName, and device properties with the exception of
// version, cannot be modified.
//
// Data sources are identified by their dataStreamId. This method
// supports patch semantics.
func (r *UsersDataSourcesService) Patch(userId string, dataSourceId string, datasource *DataSource) *UsersDataSourcesPatchCall {
	c := &UsersDataSourcesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.dataSourceId = dataSourceId
	c.datasource = datasource
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesPatchCall) Fields(s ...googleapi.Field) *UsersDataSourcesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesPatchCall) Context(ctx context.Context) *UsersDataSourcesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datasource)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources/{dataSourceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":       c.userId,
		"dataSourceId": c.dataSourceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.patch" call.
// Exactly one of *DataSource or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DataSource.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersDataSourcesPatchCall) Do(opts ...googleapi.CallOption) (*DataSource, error) {
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
	ret := &DataSource{
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
	//   "description": "Updates the specified data source. The dataStreamId, dataType, type, dataStreamName, and device properties with the exception of version, cannot be modified.\n\nData sources are identified by their dataStreamId. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "fitness.users.dataSources.patch",
	//   "parameterOrder": [
	//     "userId",
	//     "dataSourceId"
	//   ],
	//   "parameters": {
	//     "dataSourceId": {
	//       "description": "The data stream ID of the data source to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Update the data source for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources/{dataSourceId}",
	//   "request": {
	//     "$ref": "DataSource"
	//   },
	//   "response": {
	//     "$ref": "DataSource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.dataSources.update":

type UsersDataSourcesUpdateCall struct {
	s            *Service
	userId       string
	dataSourceId string
	datasource   *DataSource
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Updates the specified data source. The dataStreamId,
// dataType, type, dataStreamName, and device properties with the
// exception of version, cannot be modified.
//
// Data sources are identified by their dataStreamId.
func (r *UsersDataSourcesService) Update(userId string, dataSourceId string, datasource *DataSource) *UsersDataSourcesUpdateCall {
	c := &UsersDataSourcesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.dataSourceId = dataSourceId
	c.datasource = datasource
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesUpdateCall) Fields(s ...googleapi.Field) *UsersDataSourcesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesUpdateCall) Context(ctx context.Context) *UsersDataSourcesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.datasource)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources/{dataSourceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":       c.userId,
		"dataSourceId": c.dataSourceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.update" call.
// Exactly one of *DataSource or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DataSource.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersDataSourcesUpdateCall) Do(opts ...googleapi.CallOption) (*DataSource, error) {
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
	ret := &DataSource{
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
	//   "description": "Updates the specified data source. The dataStreamId, dataType, type, dataStreamName, and device properties with the exception of version, cannot be modified.\n\nData sources are identified by their dataStreamId.",
	//   "httpMethod": "PUT",
	//   "id": "fitness.users.dataSources.update",
	//   "parameterOrder": [
	//     "userId",
	//     "dataSourceId"
	//   ],
	//   "parameters": {
	//     "dataSourceId": {
	//       "description": "The data stream ID of the data source to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Update the data source for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources/{dataSourceId}",
	//   "request": {
	//     "$ref": "DataSource"
	//   },
	//   "response": {
	//     "$ref": "DataSource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.dataSources.datasets.delete":

type UsersDataSourcesDatasetsDeleteCall struct {
	s            *Service
	userId       string
	dataSourceId string
	datasetId    string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Performs an inclusive delete of all data points whose start
// and end times have any overlap with the time range specified by the
// dataset ID. For most data types, the entire data point will be
// deleted. For data types where the time span represents a consistent
// value (such as com.google.activity.segment), and a data point
// straddles either end point of the dataset, only the overlapping
// portion of the data point will be deleted.
func (r *UsersDataSourcesDatasetsService) Delete(userId string, dataSourceId string, datasetId string) *UsersDataSourcesDatasetsDeleteCall {
	c := &UsersDataSourcesDatasetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.dataSourceId = dataSourceId
	c.datasetId = datasetId
	return c
}

// CurrentTimeMillis sets the optional parameter "currentTimeMillis":
// The client's current time in milliseconds since epoch.
func (c *UsersDataSourcesDatasetsDeleteCall) CurrentTimeMillis(currentTimeMillis int64) *UsersDataSourcesDatasetsDeleteCall {
	c.urlParams_.Set("currentTimeMillis", fmt.Sprint(currentTimeMillis))
	return c
}

// ModifiedTimeMillis sets the optional parameter "modifiedTimeMillis":
// When the operation was performed on the client.
func (c *UsersDataSourcesDatasetsDeleteCall) ModifiedTimeMillis(modifiedTimeMillis int64) *UsersDataSourcesDatasetsDeleteCall {
	c.urlParams_.Set("modifiedTimeMillis", fmt.Sprint(modifiedTimeMillis))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesDatasetsDeleteCall) Fields(s ...googleapi.Field) *UsersDataSourcesDatasetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesDatasetsDeleteCall) Context(ctx context.Context) *UsersDataSourcesDatasetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesDatasetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesDatasetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources/{dataSourceId}/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":       c.userId,
		"dataSourceId": c.dataSourceId,
		"datasetId":    c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.datasets.delete" call.
func (c *UsersDataSourcesDatasetsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Performs an inclusive delete of all data points whose start and end times have any overlap with the time range specified by the dataset ID. For most data types, the entire data point will be deleted. For data types where the time span represents a consistent value (such as com.google.activity.segment), and a data point straddles either end point of the dataset, only the overlapping portion of the data point will be deleted.",
	//   "httpMethod": "DELETE",
	//   "id": "fitness.users.dataSources.datasets.delete",
	//   "parameterOrder": [
	//     "userId",
	//     "dataSourceId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "currentTimeMillis": {
	//       "description": "The client's current time in milliseconds since epoch.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "dataSourceId": {
	//       "description": "The data stream ID of the data source that created the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "datasetId": {
	//       "description": "Dataset identifier that is a composite of the minimum data point start time and maximum data point end time represented as nanoseconds from the epoch. The ID is formatted like: \"startTime-endTime\" where startTime and endTime are 64 bit integers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "modifiedTimeMillis": {
	//       "description": "When the operation was performed on the client.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Delete a dataset for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources/{dataSourceId}/datasets/{datasetId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.dataSources.datasets.get":

type UsersDataSourcesDatasetsGetCall struct {
	s            *Service
	userId       string
	dataSourceId string
	datasetId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns a dataset containing all data points whose start and end
// times overlap with the specified range of the dataset minimum start
// time and maximum end time. Specifically, any data point whose start
// time is less than or equal to the dataset end time and whose end time
// is greater than or equal to the dataset start time.
func (r *UsersDataSourcesDatasetsService) Get(userId string, dataSourceId string, datasetId string) *UsersDataSourcesDatasetsGetCall {
	c := &UsersDataSourcesDatasetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.dataSourceId = dataSourceId
	c.datasetId = datasetId
	return c
}

// Limit sets the optional parameter "limit": If specified, no more than
// this many data points will be included in the dataset. If there are
// more data points in the dataset, nextPageToken will be set in the
// dataset response.
func (c *UsersDataSourcesDatasetsGetCall) Limit(limit int64) *UsersDataSourcesDatasetsGetCall {
	c.urlParams_.Set("limit", fmt.Sprint(limit))
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large datasets. To get the next
// page of a dataset, set this parameter to the value of nextPageToken
// from the previous response. Each subsequent call will yield a partial
// dataset with data point end timestamps that are strictly smaller than
// those in the previous partial response.
func (c *UsersDataSourcesDatasetsGetCall) PageToken(pageToken string) *UsersDataSourcesDatasetsGetCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesDatasetsGetCall) Fields(s ...googleapi.Field) *UsersDataSourcesDatasetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersDataSourcesDatasetsGetCall) IfNoneMatch(entityTag string) *UsersDataSourcesDatasetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesDatasetsGetCall) Context(ctx context.Context) *UsersDataSourcesDatasetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesDatasetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesDatasetsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources/{dataSourceId}/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":       c.userId,
		"dataSourceId": c.dataSourceId,
		"datasetId":    c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.datasets.get" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersDataSourcesDatasetsGetCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
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
	ret := &Dataset{
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
	//   "description": "Returns a dataset containing all data points whose start and end times overlap with the specified range of the dataset minimum start time and maximum end time. Specifically, any data point whose start time is less than or equal to the dataset end time and whose end time is greater than or equal to the dataset start time.",
	//   "httpMethod": "GET",
	//   "id": "fitness.users.dataSources.datasets.get",
	//   "parameterOrder": [
	//     "userId",
	//     "dataSourceId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "dataSourceId": {
	//       "description": "The data stream ID of the data source that created the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "datasetId": {
	//       "description": "Dataset identifier that is a composite of the minimum data point start time and maximum data point end time represented as nanoseconds from the epoch. The ID is formatted like: \"startTime-endTime\" where startTime and endTime are 64 bit integers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "limit": {
	//       "description": "If specified, no more than this many data points will be included in the dataset. If there are more data points in the dataset, nextPageToken will be set in the dataset response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large datasets. To get the next page of a dataset, set this parameter to the value of nextPageToken from the previous response. Each subsequent call will yield a partial dataset with data point end timestamps that are strictly smaller than those in the previous partial response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Retrieve a dataset for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources/{dataSourceId}/datasets/{datasetId}",
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.read",
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.read",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.read",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.read",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.read",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.read",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.read",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.read",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *UsersDataSourcesDatasetsGetCall) Pages(ctx context.Context, f func(*Dataset) error) error {
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

// method id "fitness.users.dataSources.datasets.patch":

type UsersDataSourcesDatasetsPatchCall struct {
	s            *Service
	userId       string
	dataSourceId string
	datasetId    string
	dataset      *Dataset
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Patch: Adds data points to a dataset. The dataset need not be
// previously created. All points within the given dataset will be
// returned with subsquent calls to retrieve this dataset. Data points
// can belong to more than one dataset. This method does not use patch
// semantics.
func (r *UsersDataSourcesDatasetsService) Patch(userId string, dataSourceId string, datasetId string, dataset *Dataset) *UsersDataSourcesDatasetsPatchCall {
	c := &UsersDataSourcesDatasetsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.dataSourceId = dataSourceId
	c.datasetId = datasetId
	c.dataset = dataset
	return c
}

// CurrentTimeMillis sets the optional parameter "currentTimeMillis":
// The client's current time in milliseconds since epoch. Note that the
// minStartTimeNs and maxEndTimeNs properties in the request body are in
// nanoseconds instead of milliseconds.
func (c *UsersDataSourcesDatasetsPatchCall) CurrentTimeMillis(currentTimeMillis int64) *UsersDataSourcesDatasetsPatchCall {
	c.urlParams_.Set("currentTimeMillis", fmt.Sprint(currentTimeMillis))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDataSourcesDatasetsPatchCall) Fields(s ...googleapi.Field) *UsersDataSourcesDatasetsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDataSourcesDatasetsPatchCall) Context(ctx context.Context) *UsersDataSourcesDatasetsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDataSourcesDatasetsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDataSourcesDatasetsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dataset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataSources/{dataSourceId}/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":       c.userId,
		"dataSourceId": c.dataSourceId,
		"datasetId":    c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataSources.datasets.patch" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersDataSourcesDatasetsPatchCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
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
	ret := &Dataset{
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
	//   "description": "Adds data points to a dataset. The dataset need not be previously created. All points within the given dataset will be returned with subsquent calls to retrieve this dataset. Data points can belong to more than one dataset. This method does not use patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "fitness.users.dataSources.datasets.patch",
	//   "parameterOrder": [
	//     "userId",
	//     "dataSourceId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "currentTimeMillis": {
	//       "description": "The client's current time in milliseconds since epoch. Note that the minStartTimeNs and maxEndTimeNs properties in the request body are in nanoseconds instead of milliseconds.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "dataSourceId": {
	//       "description": "The data stream ID of the data source that created the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "datasetId": {
	//       "description": "Dataset identifier that is a composite of the minimum data point start time and maximum data point end time represented as nanoseconds from the epoch. The ID is formatted like: \"startTime-endTime\" where startTime and endTime are 64 bit integers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Patch a dataset for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataSources/{dataSourceId}/datasets/{datasetId}",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *UsersDataSourcesDatasetsPatchCall) Pages(ctx context.Context, f func(*Dataset) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.dataset.NextPageToken = pt }(c.dataset.NextPageToken) // reset paging to original point
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
		c.dataset.NextPageToken = x.NextPageToken
	}
}

// method id "fitness.users.dataset.aggregate":

type UsersDatasetAggregateCall struct {
	s                *Service
	userId           string
	aggregaterequest *AggregateRequest
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Aggregate: Aggregates data of a certain type or stream into buckets
// divided by a given type of boundary. Multiple data sets of multiple
// types and from multiple sources can be aggreated into exactly one
// bucket type per request.
func (r *UsersDatasetService) Aggregate(userId string, aggregaterequest *AggregateRequest) *UsersDatasetAggregateCall {
	c := &UsersDatasetAggregateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.aggregaterequest = aggregaterequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDatasetAggregateCall) Fields(s ...googleapi.Field) *UsersDatasetAggregateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDatasetAggregateCall) Context(ctx context.Context) *UsersDatasetAggregateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDatasetAggregateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDatasetAggregateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.aggregaterequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/dataset:aggregate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.dataset.aggregate" call.
// Exactly one of *AggregateResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AggregateResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersDatasetAggregateCall) Do(opts ...googleapi.CallOption) (*AggregateResponse, error) {
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
	ret := &AggregateResponse{
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
	//   "description": "Aggregates data of a certain type or stream into buckets divided by a given type of boundary. Multiple data sets of multiple types and from multiple sources can be aggreated into exactly one bucket type per request.",
	//   "httpMethod": "POST",
	//   "id": "fitness.users.dataset.aggregate",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "Aggregate data for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/dataset:aggregate",
	//   "request": {
	//     "$ref": "AggregateRequest"
	//   },
	//   "response": {
	//     "$ref": "AggregateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.read",
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.read",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.read",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.read",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.read",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.read",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.read",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.read",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// method id "fitness.users.sessions.delete":

type UsersSessionsDeleteCall struct {
	s          *Service
	userId     string
	sessionId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a session specified by the given session ID.
func (r *UsersSessionsService) Delete(userId string, sessionId string) *UsersSessionsDeleteCall {
	c := &UsersSessionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.sessionId = sessionId
	return c
}

// CurrentTimeMillis sets the optional parameter "currentTimeMillis":
// The client's current time in milliseconds since epoch.
func (c *UsersSessionsDeleteCall) CurrentTimeMillis(currentTimeMillis int64) *UsersSessionsDeleteCall {
	c.urlParams_.Set("currentTimeMillis", fmt.Sprint(currentTimeMillis))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersSessionsDeleteCall) Fields(s ...googleapi.Field) *UsersSessionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersSessionsDeleteCall) Context(ctx context.Context) *UsersSessionsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersSessionsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersSessionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/sessions/{sessionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":    c.userId,
		"sessionId": c.sessionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.sessions.delete" call.
func (c *UsersSessionsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a session specified by the given session ID.",
	//   "httpMethod": "DELETE",
	//   "id": "fitness.users.sessions.delete",
	//   "parameterOrder": [
	//     "userId",
	//     "sessionId"
	//   ],
	//   "parameters": {
	//     "currentTimeMillis": {
	//       "description": "The client's current time in milliseconds since epoch.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sessionId": {
	//       "description": "The ID of the session to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Delete a session for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/sessions/{sessionId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write"
	//   ]
	// }

}

// method id "fitness.users.sessions.list":

type UsersSessionsListCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists sessions previously created.
func (r *UsersSessionsService) List(userId string) *UsersSessionsListCall {
	c := &UsersSessionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// EndTime sets the optional parameter "endTime": An RFC3339 timestamp.
// Only sessions ending between the start and end times will be included
// in the response.
func (c *UsersSessionsListCall) EndTime(endTime string) *UsersSessionsListCall {
	c.urlParams_.Set("endTime", endTime)
	return c
}

// IncludeDeleted sets the optional parameter "includeDeleted": If true,
// deleted sessions will be returned. When set to true, sessions
// returned in this response will only have an ID and will not have any
// other fields.
func (c *UsersSessionsListCall) IncludeDeleted(includeDeleted bool) *UsersSessionsListCall {
	c.urlParams_.Set("includeDeleted", fmt.Sprint(includeDeleted))
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// nextPageToken from the previous response.
func (c *UsersSessionsListCall) PageToken(pageToken string) *UsersSessionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// StartTime sets the optional parameter "startTime": An RFC3339
// timestamp. Only sessions ending between the start and end times will
// be included in the response.
func (c *UsersSessionsListCall) StartTime(startTime string) *UsersSessionsListCall {
	c.urlParams_.Set("startTime", startTime)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersSessionsListCall) Fields(s ...googleapi.Field) *UsersSessionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersSessionsListCall) IfNoneMatch(entityTag string) *UsersSessionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersSessionsListCall) Context(ctx context.Context) *UsersSessionsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersSessionsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersSessionsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/sessions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.sessions.list" call.
// Exactly one of *ListSessionsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListSessionsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersSessionsListCall) Do(opts ...googleapi.CallOption) (*ListSessionsResponse, error) {
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
	ret := &ListSessionsResponse{
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
	//   "description": "Lists sessions previously created.",
	//   "httpMethod": "GET",
	//   "id": "fitness.users.sessions.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "endTime": {
	//       "description": "An RFC3339 timestamp. Only sessions ending between the start and end times will be included in the response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "includeDeleted": {
	//       "description": "If true, deleted sessions will be returned. When set to true, sessions returned in this response will only have an ID and will not have any other fields.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of nextPageToken from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startTime": {
	//       "description": "An RFC3339 timestamp. Only sessions ending between the start and end times will be included in the response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "List sessions for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/sessions",
	//   "response": {
	//     "$ref": "ListSessionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.read",
	//     "https://www.googleapis.com/auth/fitness.activity.write",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.read",
	//     "https://www.googleapis.com/auth/fitness.blood_glucose.write",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.read",
	//     "https://www.googleapis.com/auth/fitness.blood_pressure.write",
	//     "https://www.googleapis.com/auth/fitness.body.read",
	//     "https://www.googleapis.com/auth/fitness.body.write",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.read",
	//     "https://www.googleapis.com/auth/fitness.body_temperature.write",
	//     "https://www.googleapis.com/auth/fitness.location.read",
	//     "https://www.googleapis.com/auth/fitness.location.write",
	//     "https://www.googleapis.com/auth/fitness.nutrition.read",
	//     "https://www.googleapis.com/auth/fitness.nutrition.write",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
	//     "https://www.googleapis.com/auth/fitness.oxygen_saturation.write",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.read",
	//     "https://www.googleapis.com/auth/fitness.reproductive_health.write"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *UsersSessionsListCall) Pages(ctx context.Context, f func(*ListSessionsResponse) error) error {
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

// method id "fitness.users.sessions.update":

type UsersSessionsUpdateCall struct {
	s          *Service
	userId     string
	sessionId  string
	session    *Session
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates or insert a given session.
func (r *UsersSessionsService) Update(userId string, sessionId string, session *Session) *UsersSessionsUpdateCall {
	c := &UsersSessionsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.sessionId = sessionId
	c.session = session
	return c
}

// CurrentTimeMillis sets the optional parameter "currentTimeMillis":
// The client's current time in milliseconds since epoch.
func (c *UsersSessionsUpdateCall) CurrentTimeMillis(currentTimeMillis int64) *UsersSessionsUpdateCall {
	c.urlParams_.Set("currentTimeMillis", fmt.Sprint(currentTimeMillis))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersSessionsUpdateCall) Fields(s ...googleapi.Field) *UsersSessionsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersSessionsUpdateCall) Context(ctx context.Context) *UsersSessionsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersSessionsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersSessionsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.session)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/sessions/{sessionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId":    c.userId,
		"sessionId": c.sessionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "fitness.users.sessions.update" call.
// Exactly one of *Session or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Session.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersSessionsUpdateCall) Do(opts ...googleapi.CallOption) (*Session, error) {
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
	ret := &Session{
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
	//   "description": "Updates or insert a given session.",
	//   "httpMethod": "PUT",
	//   "id": "fitness.users.sessions.update",
	//   "parameterOrder": [
	//     "userId",
	//     "sessionId"
	//   ],
	//   "parameters": {
	//     "currentTimeMillis": {
	//       "description": "The client's current time in milliseconds since epoch.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sessionId": {
	//       "description": "The ID of the session to be created.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Create sessions for the person identified. Use me to indicate the authenticated user. Only me is supported at this time.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/sessions/{sessionId}",
	//   "request": {
	//     "$ref": "Session"
	//   },
	//   "response": {
	//     "$ref": "Session"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/fitness.activity.write"
	//   ]
	// }

}
