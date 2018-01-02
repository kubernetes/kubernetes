// Package dlp provides access to the DLP API.
//
// See https://cloud.google.com/dlp/docs/
//
// Usage example:
//
//   import "google.golang.org/api/dlp/v2beta1"
//   ...
//   dlpService, err := dlp.New(oauthHttpClient)
package dlp // import "google.golang.org/api/dlp/v2beta1"

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

const apiId = "dlp:v2beta1"
const apiName = "dlp"
const apiVersion = "v2beta1"
const basePath = "https://dlp.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Content = NewContentService(s)
	s.DataSource = NewDataSourceService(s)
	s.Inspect = NewInspectService(s)
	s.RiskAnalysis = NewRiskAnalysisService(s)
	s.RootCategories = NewRootCategoriesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Content *ContentService

	DataSource *DataSourceService

	Inspect *InspectService

	RiskAnalysis *RiskAnalysisService

	RootCategories *RootCategoriesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewContentService(s *Service) *ContentService {
	rs := &ContentService{s: s}
	return rs
}

type ContentService struct {
	s *Service
}

func NewDataSourceService(s *Service) *DataSourceService {
	rs := &DataSourceService{s: s}
	return rs
}

type DataSourceService struct {
	s *Service
}

func NewInspectService(s *Service) *InspectService {
	rs := &InspectService{s: s}
	rs.Operations = NewInspectOperationsService(s)
	rs.Results = NewInspectResultsService(s)
	return rs
}

type InspectService struct {
	s *Service

	Operations *InspectOperationsService

	Results *InspectResultsService
}

func NewInspectOperationsService(s *Service) *InspectOperationsService {
	rs := &InspectOperationsService{s: s}
	return rs
}

type InspectOperationsService struct {
	s *Service
}

func NewInspectResultsService(s *Service) *InspectResultsService {
	rs := &InspectResultsService{s: s}
	rs.Findings = NewInspectResultsFindingsService(s)
	return rs
}

type InspectResultsService struct {
	s *Service

	Findings *InspectResultsFindingsService
}

func NewInspectResultsFindingsService(s *Service) *InspectResultsFindingsService {
	rs := &InspectResultsFindingsService{s: s}
	return rs
}

type InspectResultsFindingsService struct {
	s *Service
}

func NewRiskAnalysisService(s *Service) *RiskAnalysisService {
	rs := &RiskAnalysisService{s: s}
	rs.Operations = NewRiskAnalysisOperationsService(s)
	return rs
}

type RiskAnalysisService struct {
	s *Service

	Operations *RiskAnalysisOperationsService
}

func NewRiskAnalysisOperationsService(s *Service) *RiskAnalysisOperationsService {
	rs := &RiskAnalysisOperationsService{s: s}
	return rs
}

type RiskAnalysisOperationsService struct {
	s *Service
}

func NewRootCategoriesService(s *Service) *RootCategoriesService {
	rs := &RootCategoriesService{s: s}
	rs.InfoTypes = NewRootCategoriesInfoTypesService(s)
	return rs
}

type RootCategoriesService struct {
	s *Service

	InfoTypes *RootCategoriesInfoTypesService
}

func NewRootCategoriesInfoTypesService(s *Service) *RootCategoriesInfoTypesService {
	rs := &RootCategoriesInfoTypesService{s: s}
	return rs
}

type RootCategoriesInfoTypesService struct {
	s *Service
}

// GoogleLongrunningCancelOperationRequest: The request message for
// Operations.CancelOperation.
type GoogleLongrunningCancelOperationRequest struct {
}

// GoogleLongrunningListOperationsResponse: The response message for
// Operations.ListOperations.
type GoogleLongrunningListOperationsResponse struct {
	// NextPageToken: The standard List next-page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Operations: A list of operations that matches the specified filter in
	// the request.
	Operations []*GoogleLongrunningOperation `json:"operations,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleLongrunningListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod GoogleLongrunningListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleLongrunningOperation: This resource represents a long-running
// operation that is the result of a
// network API call.
type GoogleLongrunningOperation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress.
	// If `true`, the operation is completed, and either `error` or
	// `response` is
	// available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure or
	// cancellation.
	Error *GoogleRpcStatus `json:"error,omitempty"`

	// Metadata: This field will contain an InspectOperationMetadata object.
	// This will always be returned with the Operation.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, The `name` should have the format of
	// `inspect/operations/<identifier>`.
	Name string `json:"name,omitempty"`

	// Response: This field will contain an InspectOperationResult object.
	Response googleapi.RawMessage `json:"response,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Done") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Done") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleLongrunningOperation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleLongrunningOperation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1AnalyzeDataSourceRiskRequest: Request for
// creating a risk analysis operation.
type GooglePrivacyDlpV2beta1AnalyzeDataSourceRiskRequest struct {
	// PrivacyMetric: Privacy metric to compute.
	PrivacyMetric *GooglePrivacyDlpV2beta1PrivacyMetric `json:"privacyMetric,omitempty"`

	// SourceTable: Input dataset to compute metrics over.
	SourceTable *GooglePrivacyDlpV2beta1BigQueryTable `json:"sourceTable,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PrivacyMetric") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PrivacyMetric") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1AnalyzeDataSourceRiskRequest) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1AnalyzeDataSourceRiskRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1BigQueryOptions: Options defining BigQuery
// table and row identifiers.
type GooglePrivacyDlpV2beta1BigQueryOptions struct {
	// IdentifyingFields: References to fields uniquely identifying rows
	// within the table.
	// Nested fields in the format, like `person.birthdate.year`, are
	// allowed.
	IdentifyingFields []*GooglePrivacyDlpV2beta1FieldId `json:"identifyingFields,omitempty"`

	// TableReference: Complete BigQuery table reference.
	TableReference *GooglePrivacyDlpV2beta1BigQueryTable `json:"tableReference,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IdentifyingFields")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "IdentifyingFields") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1BigQueryOptions) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1BigQueryOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1BigQueryTable: Message defining the location
// of a BigQuery table. A table is uniquely
// identified  by its project_id, dataset_id, and table_name. Within a
// query
// a table is often referenced with a string in the format
// of:
// `<project_id>:<dataset_id>.<table_id>`
// or
// `<project_id>.<dataset_id>.<table_id>`.
type GooglePrivacyDlpV2beta1BigQueryTable struct {
	// DatasetId: Dataset ID of the table.
	DatasetId string `json:"datasetId,omitempty"`

	// ProjectId: The Google Cloud Platform project ID of the project
	// containing the table.
	// If omitted, project ID is inferred from the API call.
	ProjectId string `json:"projectId,omitempty"`

	// TableId: Name of the table.
	TableId string `json:"tableId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1BigQueryTable) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1BigQueryTable
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Bucket: Buckets represented as ranges, along
// with replacement values. Ranges must
// be non-overlapping.
type GooglePrivacyDlpV2beta1Bucket struct {
	// Max: Upper bound of the range, exclusive; type must match min.
	Max *GooglePrivacyDlpV2beta1Value `json:"max,omitempty"`

	// Min: Lower bound of the range, inclusive. Type should be the same as
	// max if
	// used.
	Min *GooglePrivacyDlpV2beta1Value `json:"min,omitempty"`

	// ReplacementValue: Replacement value for this bucket. If not
	// provided
	// the default behavior will be to hyphenate the min-max range.
	ReplacementValue *GooglePrivacyDlpV2beta1Value `json:"replacementValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Max") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Max") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Bucket) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Bucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1BucketingConfig: Generalization function that
// buckets values based on ranges. The ranges and
// replacement values are dynamically provided by the user for custom
// behavior,
// such as 1-30 -> LOW 31-65 -> MEDIUM 66-100 -> HIGH
// This can be used on
// data of type: number, long, string, timestamp.
// If the bound `Value` type differs from the type of data being
// transformed, we
// will first attempt converting the type of the data to be transformed
// to match
// the type of the bound before comparing.
type GooglePrivacyDlpV2beta1BucketingConfig struct {
	Buckets []*GooglePrivacyDlpV2beta1Bucket `json:"buckets,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1BucketingConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1BucketingConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CategoricalStatsConfig: Compute numerical
// stats over an individual column, including
// number of distinct values and value count distribution.
type GooglePrivacyDlpV2beta1CategoricalStatsConfig struct {
	// Field: Field to compute categorical stats on. All column types
	// are
	// supported except for arrays and structs. However, it may be
	// more
	// informative to use NumericalStats when the field type is
	// supported,
	// depending on the data.
	Field *GooglePrivacyDlpV2beta1FieldId `json:"field,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1CategoricalStatsConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CategoricalStatsConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GooglePrivacyDlpV2beta1CategoricalStatsHistogramBucket struct {
	// BucketSize: Total number of records in this bucket.
	BucketSize int64 `json:"bucketSize,omitempty,string"`

	// BucketValues: Sample of value frequencies in this bucket. The total
	// number of
	// values returned per bucket is capped at 20.
	BucketValues []*GooglePrivacyDlpV2beta1ValueFrequency `json:"bucketValues,omitempty"`

	// ValueFrequencyLowerBound: Lower bound on the value frequency of the
	// values in this bucket.
	ValueFrequencyLowerBound int64 `json:"valueFrequencyLowerBound,omitempty,string"`

	// ValueFrequencyUpperBound: Upper bound on the value frequency of the
	// values in this bucket.
	ValueFrequencyUpperBound int64 `json:"valueFrequencyUpperBound,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "BucketSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketSize") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CategoricalStatsHistogramBucket) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CategoricalStatsHistogramBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CategoricalStatsResult: Result of the
// categorical stats computation.
type GooglePrivacyDlpV2beta1CategoricalStatsResult struct {
	// ValueFrequencyHistogramBuckets: Histogram of value frequencies in the
	// column.
	ValueFrequencyHistogramBuckets []*GooglePrivacyDlpV2beta1CategoricalStatsHistogramBucket `json:"valueFrequencyHistogramBuckets,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ValueFrequencyHistogramBuckets") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "ValueFrequencyHistogramBuckets") to include in API requests with the
	// JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CategoricalStatsResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CategoricalStatsResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CategoryDescription: Info Type Category
// description.
type GooglePrivacyDlpV2beta1CategoryDescription struct {
	// DisplayName: Human readable form of the category name.
	DisplayName string `json:"displayName,omitempty"`

	// Name: Internal name of the category.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CategoryDescription) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CategoryDescription
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CharacterMaskConfig: Partially mask a string
// by replacing a given number of characters with a
// fixed character. Masking can start from the beginning or end of the
// string.
// This can be used on data of any type (numbers, longs, and so on) and
// when
// de-identifying structured data we'll attempt to preserve the original
// data's
// type. (This allows you to take a long like 123 and modify it to a
// string like
// **3.
type GooglePrivacyDlpV2beta1CharacterMaskConfig struct {
	// CharactersToIgnore: When masking a string, items in this list will be
	// skipped when replacing.
	// For example, if your string is 555-555-5555 and you ask us to skip
	// `-` and
	// mask 5 chars with * we would produce ***-*55-5555.
	CharactersToIgnore []*GooglePrivacyDlpV2beta1CharsToIgnore `json:"charactersToIgnore,omitempty"`

	// MaskingCharacter: Character to mask the sensitive values&mdash;for
	// example, "*" for an
	// alphabetic string such as name, or "0" for a numeric string such as
	// ZIP
	// code or credit card number. String must have length 1. If not
	// supplied, we
	// will default to "*" for strings, 0 for digits.
	MaskingCharacter string `json:"maskingCharacter,omitempty"`

	// NumberToMask: Number of characters to mask. If not set, all matching
	// chars will be
	// masked. Skipped characters do not count towards this tally.
	NumberToMask int64 `json:"numberToMask,omitempty"`

	// ReverseOrder: Mask characters in reverse order. For example, if
	// `masking_character` is
	// '0', number_to_mask is 14, and `reverse_order` is false,
	// then
	// 1234-5678-9012-3456 -> 00000000000000-3456
	// If `masking_character` is '*', `number_to_mask` is 3, and
	// `reverse_order`
	// is true, then 12345 -> 12***
	ReverseOrder bool `json:"reverseOrder,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CharactersToIgnore")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CharactersToIgnore") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CharacterMaskConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CharacterMaskConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CharsToIgnore: Characters to skip when doing
// deidentification of a value. These will be left
// alone and skipped.
type GooglePrivacyDlpV2beta1CharsToIgnore struct {
	CharactersToSkip string `json:"charactersToSkip,omitempty"`

	// Possible values:
	//   "CHARACTER_GROUP_UNSPECIFIED"
	//   "NUMERIC" - 0-9
	//   "ALPHA_UPPER_CASE" - A-Z
	//   "ALPHA_LOWER_CASE" - a-z
	//   "PUNCTUATION" - US Punctuation, one of
	// !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
	//   "WHITESPACE" - Whitespace character, one of [ \t\n\x0B\f\r]
	CommonCharactersToIgnore string `json:"commonCharactersToIgnore,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CharactersToSkip") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CharactersToSkip") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CharsToIgnore) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CharsToIgnore
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CloudStorageKey: Record key for a finding in a
// Cloud Storage file.
type GooglePrivacyDlpV2beta1CloudStorageKey struct {
	// FilePath: Path to the file.
	FilePath string `json:"filePath,omitempty"`

	// StartOffset: Byte offset of the referenced data in the file.
	StartOffset int64 `json:"startOffset,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "FilePath") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FilePath") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CloudStorageKey) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CloudStorageKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CloudStorageOptions: Options defining a file
// or a set of files (path ending with *) within
// a Google Cloud Storage bucket.
type GooglePrivacyDlpV2beta1CloudStorageOptions struct {
	FileSet *GooglePrivacyDlpV2beta1FileSet `json:"fileSet,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FileSet") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FileSet") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CloudStorageOptions) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CloudStorageOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CloudStoragePath: A location in Cloud Storage.
type GooglePrivacyDlpV2beta1CloudStoragePath struct {
	// Path: The url, in the format of `gs://bucket/<path>`.
	Path string `json:"path,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Path") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Path") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CloudStoragePath) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CloudStoragePath
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Color: Represents a color in the RGB color
// space.
type GooglePrivacyDlpV2beta1Color struct {
	// Blue: The amount of blue in the color as a value in the interval [0,
	// 1].
	Blue float64 `json:"blue,omitempty"`

	// Green: The amount of green in the color as a value in the interval
	// [0, 1].
	Green float64 `json:"green,omitempty"`

	// Red: The amount of red in the color as a value in the interval [0,
	// 1].
	Red float64 `json:"red,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Blue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Blue") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Color) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Color
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *GooglePrivacyDlpV2beta1Color) UnmarshalJSON(data []byte) error {
	type noMethod GooglePrivacyDlpV2beta1Color
	var s1 struct {
		Blue  gensupport.JSONFloat64 `json:"blue"`
		Green gensupport.JSONFloat64 `json:"green"`
		Red   gensupport.JSONFloat64 `json:"red"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Blue = float64(s1.Blue)
	s.Green = float64(s1.Green)
	s.Red = float64(s1.Red)
	return nil
}

// GooglePrivacyDlpV2beta1Condition: The field type of `value` and
// `field` do not need to match to be
// considered equal, but not all comparisons are possible.
//
// A `value` of type:
//
// - `string` can be compared against all other types
// - `boolean` can only be compared against other booleans
// - `integer` can be compared against doubles or a string if the string
// value
// can be parsed as an integer.
// - `double` can be compared against integers or a string if the string
// can
// be parsed as a double.
// - `Timestamp` can be compared against strings in RFC 3339 date
// string
// format.
// - `TimeOfDay` can be compared against timestamps and strings in the
// format
// of 'HH:mm:ss'.
//
// If we fail to compare do to type mismatch, a warning will be given
// and
// the condition will evaluate to false.
type GooglePrivacyDlpV2beta1Condition struct {
	// Field: Field within the record this condition is evaluated against.
	// [required]
	Field *GooglePrivacyDlpV2beta1FieldId `json:"field,omitempty"`

	// Operator: Operator used to compare the field or info type to the
	// value. [required]
	//
	// Possible values:
	//   "RELATIONAL_OPERATOR_UNSPECIFIED"
	//   "EQUAL_TO" - Equal.
	//   "NOT_EQUAL_TO" - Not equal to.
	//   "GREATER_THAN" - Greater than.
	//   "LESS_THAN" - Less than.
	//   "GREATER_THAN_OR_EQUALS" - Greater than or equals.
	//   "LESS_THAN_OR_EQUALS" - Less than or equals.
	//   "EXISTS" - Exists
	Operator string `json:"operator,omitempty"`

	// Value: Value to compare against. [Required, except for `EXISTS`
	// tests.]
	Value *GooglePrivacyDlpV2beta1Value `json:"value,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1Condition) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Condition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GooglePrivacyDlpV2beta1Conditions struct {
	Conditions []*GooglePrivacyDlpV2beta1Condition `json:"conditions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Conditions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Conditions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Conditions) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Conditions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1ContentItem: Container structure for the
// content to inspect.
type GooglePrivacyDlpV2beta1ContentItem struct {
	// Data: Content data to inspect or redact.
	Data string `json:"data,omitempty"`

	// Table: Structured content for inspection.
	Table *GooglePrivacyDlpV2beta1Table `json:"table,omitempty"`

	// Type: Type of the content, as defined in Content-Type HTTP
	// header.
	// Supported types are: all "text" types, octet streams, PNG
	// images,
	// JPEG images.
	Type string `json:"type,omitempty"`

	// Value: String data to inspect or redact.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Data") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ContentItem) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ContentItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CreateInspectOperationRequest: Request for
// scheduling a scan of a data subset from a Google Platform
// data
// repository.
type GooglePrivacyDlpV2beta1CreateInspectOperationRequest struct {
	// InspectConfig: Configuration for the inspector.
	InspectConfig *GooglePrivacyDlpV2beta1InspectConfig `json:"inspectConfig,omitempty"`

	// OperationConfig: Additional configuration settings for long running
	// operations.
	OperationConfig *GooglePrivacyDlpV2beta1OperationConfig `json:"operationConfig,omitempty"`

	// OutputConfig: Optional location to store findings. The bucket must
	// already exist and
	// the Google APIs service account for DLP must have write permission
	// to
	// write to the given bucket.
	// Results are split over multiple csv files with each file name
	// matching
	// the pattern "[operation_id]_[count].csv", for
	// example
	// `3094877188788974909_1.csv`. The `operation_id` matches
	// the
	// identifier for the Operation, and the `count` is a counter used
	// for
	// tracking the number of files written.
	//
	// The CSV file(s) contain the following columns regardless of storage
	// type
	// scanned:
	// - id
	// - info_type
	// - likelihood
	// - byte size of finding
	// - quote
	// - timestamp
	//
	// For Cloud Storage the next columns are:
	//
	// - file_path
	// - start_offset
	//
	// For Cloud Datastore the next columns are:
	//
	// - project_id
	// - namespace_id
	// - path
	// - column_name
	// - offset
	//
	// For BigQuery the next columns are:
	//
	// - row_number
	// - project_id
	// - dataset_id
	// - table_id
	OutputConfig *GooglePrivacyDlpV2beta1OutputStorageConfig `json:"outputConfig,omitempty"`

	// StorageConfig: Specification of the data set to process.
	StorageConfig *GooglePrivacyDlpV2beta1StorageConfig `json:"storageConfig,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InspectConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InspectConfig") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CreateInspectOperationRequest) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CreateInspectOperationRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CryptoHashConfig: Pseudonymization method that
// generates surrogates via cryptographic hashing.
// Uses SHA-256.
// The key size must be either 32 or 64 bytes.
// Outputs a 32 byte digest as an uppercase hex string
// (for example, 41D1567F7F99F1DC2A5FAB886DEE5BEE).
// Currently, only string and integer values can be hashed.
type GooglePrivacyDlpV2beta1CryptoHashConfig struct {
	// CryptoKey: The key used by the hash function.
	CryptoKey *GooglePrivacyDlpV2beta1CryptoKey `json:"cryptoKey,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CryptoKey") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CryptoKey") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CryptoHashConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CryptoHashConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CryptoKey: This is a data encryption key (DEK)
// (as opposed to
// a key encryption key (KEK) stored by KMS).
// When using KMS to wrap/unwrap DEKs, be sure to set an appropriate
// IAM policy on the KMS CryptoKey (KEK) to ensure an attacker
// cannot
// unwrap the data crypto key.
type GooglePrivacyDlpV2beta1CryptoKey struct {
	KmsWrapped *GooglePrivacyDlpV2beta1KmsWrappedCryptoKey `json:"kmsWrapped,omitempty"`

	Transient *GooglePrivacyDlpV2beta1TransientCryptoKey `json:"transient,omitempty"`

	Unwrapped *GooglePrivacyDlpV2beta1UnwrappedCryptoKey `json:"unwrapped,omitempty"`

	// ForceSendFields is a list of field names (e.g. "KmsWrapped") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "KmsWrapped") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CryptoKey) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CryptoKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1CryptoReplaceFfxFpeConfig: Replaces an
// identifier with an surrogate using FPE with the FFX
// mode of operation.
// The identifier must be encoded as ASCII.
// For a given crypto key and context, the same identifier will
// be
// replaced with the same surrogate.
// Note that a given identifier must be either the empty string or be
// at
// least two characters long.
type GooglePrivacyDlpV2beta1CryptoReplaceFfxFpeConfig struct {
	// Possible values:
	//   "FFX_COMMON_NATIVE_ALPHABET_UNSPECIFIED"
	//   "NUMERIC" - [0-9] (radix of 10)
	//   "HEXADECIMAL" - [0-9A-F] (radix of 16)
	//   "UPPER_CASE_ALPHA_NUMERIC" - [0-9A-Z] (radix of 36)
	//   "ALPHA_NUMERIC" - [0-9A-Za-z] (radix of 62)
	CommonAlphabet string `json:"commonAlphabet,omitempty"`

	// Context: The 'tweak', a context may be used for higher security since
	// the same
	// identifier in two different contexts won't be given the same
	// surrogate. If
	// the context is not set, a default tweak will be used.
	//
	// If the context is set but:
	//
	// 1. there is no record present when transforming a given value or
	// 1. the field is not present when transforming a given value,
	//
	// a default tweak will be used.
	//
	// Note that case (1) is expected when an `InfoTypeTransformation`
	// is
	// applied to both structured and non-structured
	// `ContentItem`s.
	// Currently, the referenced field may be of value type integer or
	// string.
	//
	// The tweak is constructed as a sequence of bytes in big endian byte
	// order
	// such that:
	//
	// - a 64 bit integer is encoded followed by a single byte of value 1
	// - a string is encoded in UTF-8 format followed by a single byte of
	// value 2
	Context *GooglePrivacyDlpV2beta1FieldId `json:"context,omitempty"`

	// CryptoKey: The key used by the encryption algorithm. [required]
	CryptoKey *GooglePrivacyDlpV2beta1CryptoKey `json:"cryptoKey,omitempty"`

	// CustomAlphabet: This is supported by mapping these to the
	// alphanumeric characters
	// that the FFX mode natively supports. This happens
	// before/after
	// encryption/decryption.
	// Each character listed must appear only once.
	// Number of characters must be in the range [2, 62].
	// This must be encoded as ASCII.
	// The order of characters does not matter.
	CustomAlphabet string `json:"customAlphabet,omitempty"`

	// Radix: The native way to select the alphabet. Must be in the range
	// [2, 62].
	Radix int64 `json:"radix,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommonAlphabet") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CommonAlphabet") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1CryptoReplaceFfxFpeConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1CryptoReplaceFfxFpeConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1DatastoreKey: Record key for a finding in
// Cloud Datastore.
type GooglePrivacyDlpV2beta1DatastoreKey struct {
	// EntityKey: Datastore entity key.
	EntityKey *GooglePrivacyDlpV2beta1Key `json:"entityKey,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EntityKey") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EntityKey") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1DatastoreKey) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1DatastoreKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1DatastoreOptions: Options defining a data set
// within Google Cloud Datastore.
type GooglePrivacyDlpV2beta1DatastoreOptions struct {
	// Kind: The kind to process.
	Kind *GooglePrivacyDlpV2beta1KindExpression `json:"kind,omitempty"`

	// PartitionId: A partition ID identifies a grouping of entities. The
	// grouping is always
	// by project and namespace, however the namespace ID may be empty.
	PartitionId *GooglePrivacyDlpV2beta1PartitionId `json:"partitionId,omitempty"`

	// Projection: Properties to scan. If none are specified, all properties
	// will be scanned
	// by default.
	Projection []*GooglePrivacyDlpV2beta1Projection `json:"projection,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1DatastoreOptions) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1DatastoreOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1DeidentificationSummary: High level summary of
// deidentification.
type GooglePrivacyDlpV2beta1DeidentificationSummary struct {
	// TransformationSummaries: Transformations applied to the dataset.
	TransformationSummaries []*GooglePrivacyDlpV2beta1TransformationSummary `json:"transformationSummaries,omitempty"`

	// TransformedBytes: Total size in bytes that were transformed in some
	// way.
	TransformedBytes int64 `json:"transformedBytes,omitempty,string"`

	// ForceSendFields is a list of field names (e.g.
	// "TransformationSummaries") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TransformationSummaries")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1DeidentificationSummary) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1DeidentificationSummary
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1DeidentifyConfig: The configuration that
// controls how the data will change.
type GooglePrivacyDlpV2beta1DeidentifyConfig struct {
	// InfoTypeTransformations: Treat the dataset as free-form text and
	// apply the same free text
	// transformation everywhere.
	InfoTypeTransformations *GooglePrivacyDlpV2beta1InfoTypeTransformations `json:"infoTypeTransformations,omitempty"`

	// RecordTransformations: Treat the dataset as structured.
	// Transformations can be applied to
	// specific locations within structured datasets, such as transforming
	// a column within a table.
	RecordTransformations *GooglePrivacyDlpV2beta1RecordTransformations `json:"recordTransformations,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "InfoTypeTransformations") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InfoTypeTransformations")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1DeidentifyConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1DeidentifyConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1DeidentifyContentRequest: Request to
// de-identify a list of items.
type GooglePrivacyDlpV2beta1DeidentifyContentRequest struct {
	// DeidentifyConfig: Configuration for the de-identification of the list
	// of content items.
	DeidentifyConfig *GooglePrivacyDlpV2beta1DeidentifyConfig `json:"deidentifyConfig,omitempty"`

	// InspectConfig: Configuration for the inspector.
	InspectConfig *GooglePrivacyDlpV2beta1InspectConfig `json:"inspectConfig,omitempty"`

	// Items: The list of items to inspect. Up to 100 are allowed per
	// request.
	// All items will be treated as text/*.
	Items []*GooglePrivacyDlpV2beta1ContentItem `json:"items,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeidentifyConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeidentifyConfig") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1DeidentifyContentRequest) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1DeidentifyContentRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1DeidentifyContentResponse: Results of
// de-identifying a list of items.
type GooglePrivacyDlpV2beta1DeidentifyContentResponse struct {
	Items []*GooglePrivacyDlpV2beta1ContentItem `json:"items,omitempty"`

	// Summaries: A review of the transformations that took place for each
	// item.
	Summaries []*GooglePrivacyDlpV2beta1DeidentificationSummary `json:"summaries,omitempty"`

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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1DeidentifyContentResponse) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1DeidentifyContentResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1EntityId: An entity in a dataset is a field or
// set of fields that correspond to a
// single person. For example, in medical records the `EntityId` might
// be
// a patient identifier, or for financial records it might be an
// account
// identifier. This message is used when generalizations or analysis
// must be
// consistent across multiple rows pertaining to the same entity.
type GooglePrivacyDlpV2beta1EntityId struct {
	// Field: Composite key indicating which field contains the entity
	// identifier.
	Field *GooglePrivacyDlpV2beta1FieldId `json:"field,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1EntityId) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1EntityId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Expressions: A collection of expressions
type GooglePrivacyDlpV2beta1Expressions struct {
	Conditions *GooglePrivacyDlpV2beta1Conditions `json:"conditions,omitempty"`

	// LogicalOperator: The operator to apply to the result of conditions.
	// Default and currently
	// only supported value is `AND`.
	//
	// Possible values:
	//   "LOGICAL_OPERATOR_UNSPECIFIED"
	//   "AND"
	LogicalOperator string `json:"logicalOperator,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Conditions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Conditions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Expressions) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Expressions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1FieldId: General identifier of a data field in
// a storage service.
type GooglePrivacyDlpV2beta1FieldId struct {
	// ColumnName: Name describing the field.
	ColumnName string `json:"columnName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1FieldId) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1FieldId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1FieldTransformation: The transformation to
// apply to the field.
type GooglePrivacyDlpV2beta1FieldTransformation struct {
	// Condition: Only apply the transformation if the condition evaluates
	// to true for the
	// given `RecordCondition`. The conditions are allowed to reference
	// fields
	// that are not used in the actual transformation. [optional]
	//
	// Example Use Cases:
	//
	// - Apply a different bucket transformation to an age column if the zip
	// code
	// column for the same record is within a specific range.
	// - Redact a field if the date of birth field is greater than 85.
	Condition *GooglePrivacyDlpV2beta1RecordCondition `json:"condition,omitempty"`

	// Fields: Input field(s) to apply the transformation to. [required]
	Fields []*GooglePrivacyDlpV2beta1FieldId `json:"fields,omitempty"`

	// InfoTypeTransformations: Treat the contents of the field as free
	// text, and selectively
	// transform content that matches an `InfoType`.
	InfoTypeTransformations *GooglePrivacyDlpV2beta1InfoTypeTransformations `json:"infoTypeTransformations,omitempty"`

	// PrimitiveTransformation: Apply the transformation to the entire
	// field.
	PrimitiveTransformation *GooglePrivacyDlpV2beta1PrimitiveTransformation `json:"primitiveTransformation,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Condition") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Condition") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1FieldTransformation) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1FieldTransformation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1FileSet: Set of files to scan.
type GooglePrivacyDlpV2beta1FileSet struct {
	// Url: The url, in the format `gs://<bucket>/<path>`. Trailing wildcard
	// in the
	// path is allowed.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Url") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Url") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1FileSet) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1FileSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Finding: Container structure describing a
// single finding within a string or image.
type GooglePrivacyDlpV2beta1Finding struct {
	// CreateTime: Timestamp when finding was detected.
	CreateTime string `json:"createTime,omitempty"`

	// InfoType: The specific type of info the string might be.
	InfoType *GooglePrivacyDlpV2beta1InfoType `json:"infoType,omitempty"`

	// Likelihood: Estimate of how likely it is that the info_type is
	// correct.
	//
	// Possible values:
	//   "LIKELIHOOD_UNSPECIFIED" - Default value; information with all
	// likelihoods is included.
	//   "VERY_UNLIKELY" - Few matching elements.
	//   "UNLIKELY"
	//   "POSSIBLE" - Some matching elements.
	//   "LIKELY"
	//   "VERY_LIKELY" - Many matching elements.
	Likelihood string `json:"likelihood,omitempty"`

	// Location: Location of the info found.
	Location *GooglePrivacyDlpV2beta1Location `json:"location,omitempty"`

	// Quote: The specific string that may be potentially sensitive info.
	Quote string `json:"quote,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Finding) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Finding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1FixedSizeBucketingConfig: Buckets values based
// on fixed size ranges. The
// Bucketing transformation can provide all of this functionality,
// but requires more configuration. This message is provided as a
// convenience to
// the user for simple bucketing strategies.
// The resulting value will be a hyphenated string
// of
// lower_bound-upper_bound.
// This can be used on data of type: double, long.
// If the bound Value type differs from the type of data
// being transformed, we will first attempt converting the type of the
// data to
// be transformed to match the type of the bound before comparing.
type GooglePrivacyDlpV2beta1FixedSizeBucketingConfig struct {
	// BucketSize: Size of each bucket (except for minimum and maximum
	// buckets). So if
	// `lower_bound` = 10, `upper_bound` = 89, and `bucket_size` = 10, then
	// the
	// following buckets would be used: -10, 10-20, 20-30, 30-40, 40-50,
	// 50-60,
	// 60-70, 70-80, 80-89, 89+. Precision up to 2 decimals works.
	// [Required].
	BucketSize float64 `json:"bucketSize,omitempty"`

	// LowerBound: Lower bound value of buckets. All values less than
	// `lower_bound` are
	// grouped together into a single bucket; for example if `lower_bound` =
	// 10,
	// then all values less than 10 are replaced with the value “-10”.
	// [Required].
	LowerBound *GooglePrivacyDlpV2beta1Value `json:"lowerBound,omitempty"`

	// UpperBound: Upper bound value of buckets. All values greater than
	// upper_bound are
	// grouped together into a single bucket; for example if `upper_bound` =
	// 89,
	// then all values greater than 89 are replaced with the value
	// “89+”.
	// [Required].
	UpperBound *GooglePrivacyDlpV2beta1Value `json:"upperBound,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BucketSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketSize") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1FixedSizeBucketingConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1FixedSizeBucketingConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *GooglePrivacyDlpV2beta1FixedSizeBucketingConfig) UnmarshalJSON(data []byte) error {
	type noMethod GooglePrivacyDlpV2beta1FixedSizeBucketingConfig
	var s1 struct {
		BucketSize gensupport.JSONFloat64 `json:"bucketSize"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.BucketSize = float64(s1.BucketSize)
	return nil
}

// GooglePrivacyDlpV2beta1ImageLocation: Bounding box encompassing
// detected text within an image.
type GooglePrivacyDlpV2beta1ImageLocation struct {
	// Height: Height of the bounding box in pixels.
	Height int64 `json:"height,omitempty"`

	// Left: Left coordinate of the bounding box. (0,0) is upper left.
	Left int64 `json:"left,omitempty"`

	// Top: Top coordinate of the bounding box. (0,0) is upper left.
	Top int64 `json:"top,omitempty"`

	// Width: Width of the bounding box in pixels.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Height") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Height") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ImageLocation) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ImageLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1ImageRedactionConfig: Configuration for
// determing how redaction of images should occur.
type GooglePrivacyDlpV2beta1ImageRedactionConfig struct {
	// InfoType: Only one per info_type should be provided per request. If
	// not
	// specified, and redact_all_text is false, the DLP API will redact
	// all
	// text that it matches against all info_types that are found, but
	// not
	// specified in another ImageRedactionConfig.
	InfoType *GooglePrivacyDlpV2beta1InfoType `json:"infoType,omitempty"`

	// RedactAllText: If true, all text found in the image, regardless
	// whether it matches an
	// info_type, is redacted.
	RedactAllText bool `json:"redactAllText,omitempty"`

	// RedactionColor: The color to use when redacting content from an
	// image. If not specified,
	// the default is black.
	RedactionColor *GooglePrivacyDlpV2beta1Color `json:"redactionColor,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InfoType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InfoType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ImageRedactionConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ImageRedactionConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InfoType: Type of information detected by the
// API.
type GooglePrivacyDlpV2beta1InfoType struct {
	// Name: Name of the information type.
	Name string `json:"name,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1InfoType) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InfoType
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InfoTypeDescription: Info type description.
type GooglePrivacyDlpV2beta1InfoTypeDescription struct {
	// Categories: List of categories this info type belongs to.
	Categories []*GooglePrivacyDlpV2beta1CategoryDescription `json:"categories,omitempty"`

	// DisplayName: Human readable form of the info type name.
	DisplayName string `json:"displayName,omitempty"`

	// Name: Internal name of the info type.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Categories") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Categories") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InfoTypeDescription) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InfoTypeDescription
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InfoTypeLimit: Max findings configuration per
// info type, per content item or long running
// operation.
type GooglePrivacyDlpV2beta1InfoTypeLimit struct {
	// InfoType: Type of information the findings limit applies to. Only one
	// limit per
	// info_type should be provided. If InfoTypeLimit does not have
	// an
	// info_type, the DLP API applies the limit against all info_types that
	// are
	// found but not specified in another InfoTypeLimit.
	InfoType *GooglePrivacyDlpV2beta1InfoType `json:"infoType,omitempty"`

	// MaxFindings: Max findings limit for the given infoType.
	MaxFindings int64 `json:"maxFindings,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InfoType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InfoType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InfoTypeLimit) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InfoTypeLimit
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InfoTypeStatistics: Statistics regarding a
// specific InfoType.
type GooglePrivacyDlpV2beta1InfoTypeStatistics struct {
	// Count: Number of findings for this info type.
	Count int64 `json:"count,omitempty,string"`

	// InfoType: The type of finding this stat is for.
	InfoType *GooglePrivacyDlpV2beta1InfoType `json:"infoType,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1InfoTypeStatistics) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InfoTypeStatistics
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InfoTypeTransformation: A transformation to
// apply to text that is identified as a specific
// info_type.
type GooglePrivacyDlpV2beta1InfoTypeTransformation struct {
	// InfoTypes: Info types to apply the transformation to. Empty list will
	// match all
	// available info types for this transformation.
	InfoTypes []*GooglePrivacyDlpV2beta1InfoType `json:"infoTypes,omitempty"`

	// PrimitiveTransformation: Primitive transformation to apply to the
	// info type. [required]
	PrimitiveTransformation *GooglePrivacyDlpV2beta1PrimitiveTransformation `json:"primitiveTransformation,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InfoTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InfoTypes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InfoTypeTransformation) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InfoTypeTransformation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InfoTypeTransformations: A type of
// transformation that will scan unstructured text and
// apply various `PrimitiveTransformation`s to each finding, where
// the
// transformation is applied to only values that were identified as a
// specific
// info_type.
type GooglePrivacyDlpV2beta1InfoTypeTransformations struct {
	// Transformations: Transformation for each info type. Cannot specify
	// more than one
	// for a given info type. [required]
	Transformations []*GooglePrivacyDlpV2beta1InfoTypeTransformation `json:"transformations,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Transformations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Transformations") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InfoTypeTransformations) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InfoTypeTransformations
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InspectConfig: Configuration description of
// the scanning process.
// When used with redactContent only info_types and min_likelihood are
// currently
// used.
type GooglePrivacyDlpV2beta1InspectConfig struct {
	// ExcludeTypes: When true, excludes type information of the findings.
	ExcludeTypes bool `json:"excludeTypes,omitempty"`

	// IncludeQuote: When true, a contextual quote from the data that
	// triggered a finding is
	// included in the response; see Finding.quote.
	IncludeQuote bool `json:"includeQuote,omitempty"`

	// InfoTypeLimits: Configuration of findings limit given for specified
	// info types.
	InfoTypeLimits []*GooglePrivacyDlpV2beta1InfoTypeLimit `json:"infoTypeLimits,omitempty"`

	// InfoTypes: Restricts what info_types to look for. The values must
	// correspond to
	// InfoType values returned by ListInfoTypes or found in
	// documentation.
	// Empty info_types runs all enabled detectors.
	InfoTypes []*GooglePrivacyDlpV2beta1InfoType `json:"infoTypes,omitempty"`

	// MaxFindings: Limits the number of findings per content item or long
	// running operation.
	MaxFindings int64 `json:"maxFindings,omitempty"`

	// MinLikelihood: Only returns findings equal or above this threshold.
	//
	// Possible values:
	//   "LIKELIHOOD_UNSPECIFIED" - Default value; information with all
	// likelihoods is included.
	//   "VERY_UNLIKELY" - Few matching elements.
	//   "UNLIKELY"
	//   "POSSIBLE" - Some matching elements.
	//   "LIKELY"
	//   "VERY_LIKELY" - Many matching elements.
	MinLikelihood string `json:"minLikelihood,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExcludeTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExcludeTypes") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InspectConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InspectConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InspectContentRequest: Request to search for
// potentially sensitive info in a list of items.
type GooglePrivacyDlpV2beta1InspectContentRequest struct {
	// InspectConfig: Configuration for the inspector.
	InspectConfig *GooglePrivacyDlpV2beta1InspectConfig `json:"inspectConfig,omitempty"`

	// Items: The list of items to inspect. Items in a single request
	// are
	// considered "related" unless inspect_config.independent_inputs is
	// true.
	// Up to 100 are allowed per request.
	Items []*GooglePrivacyDlpV2beta1ContentItem `json:"items,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InspectConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InspectConfig") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InspectContentRequest) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InspectContentRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InspectContentResponse: Results of inspecting
// a list of items.
type GooglePrivacyDlpV2beta1InspectContentResponse struct {
	// Results: Each content_item from the request has a result in this
	// list, in the
	// same order as the request.
	Results []*GooglePrivacyDlpV2beta1InspectResult `json:"results,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Results") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Results") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InspectContentResponse) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InspectContentResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InspectOperationMetadata: Metadata returned
// within GetOperation for an inspect request.
type GooglePrivacyDlpV2beta1InspectOperationMetadata struct {
	// CreateTime: The time which this request was started.
	CreateTime string `json:"createTime,omitempty"`

	InfoTypeStats []*GooglePrivacyDlpV2beta1InfoTypeStatistics `json:"infoTypeStats,omitempty"`

	// ProcessedBytes: Total size in bytes that were processed.
	ProcessedBytes int64 `json:"processedBytes,omitempty,string"`

	// RequestInspectConfig: The inspect config used to create the
	// Operation.
	RequestInspectConfig *GooglePrivacyDlpV2beta1InspectConfig `json:"requestInspectConfig,omitempty"`

	// RequestOutputConfig: Optional location to store findings.
	RequestOutputConfig *GooglePrivacyDlpV2beta1OutputStorageConfig `json:"requestOutputConfig,omitempty"`

	// RequestStorageConfig: The storage config used to create the
	// Operation.
	RequestStorageConfig *GooglePrivacyDlpV2beta1StorageConfig `json:"requestStorageConfig,omitempty"`

	// TotalEstimatedBytes: Estimate of the number of bytes to process.
	TotalEstimatedBytes int64 `json:"totalEstimatedBytes,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InspectOperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InspectOperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InspectOperationResult: The operational data.
type GooglePrivacyDlpV2beta1InspectOperationResult struct {
	// Name: The server-assigned name, which is only unique within the same
	// service that
	// originally returns it. If you use the default HTTP mapping,
	// the
	// `name` should have the format of `inspect/results/{id}`.
	Name string `json:"name,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1InspectOperationResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InspectOperationResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1InspectResult: All the findings for a single
// scanned item.
type GooglePrivacyDlpV2beta1InspectResult struct {
	// Findings: List of findings for an item.
	Findings []*GooglePrivacyDlpV2beta1Finding `json:"findings,omitempty"`

	// FindingsTruncated: If true, then this item might have more findings
	// than were returned,
	// and the findings returned are an arbitrary subset of all
	// findings.
	// The findings list might be truncated because the input items were
	// too
	// large, or because the server reached the maximum amount of
	// resources
	// allowed for a single API call. For best results, divide the input
	// into
	// smaller batches.
	FindingsTruncated bool `json:"findingsTruncated,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Findings") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Findings") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1InspectResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1InspectResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1KAnonymityConfig: k-anonymity metric, used for
// analysis of reidentification risk.
type GooglePrivacyDlpV2beta1KAnonymityConfig struct {
	// EntityId: Optional message indicating that each distinct `EntityId`
	// should not
	// contribute to the k-anonymity count more than once per equivalence
	// class.
	EntityId *GooglePrivacyDlpV2beta1EntityId `json:"entityId,omitempty"`

	// QuasiIds: Set of fields to compute k-anonymity over. When multiple
	// fields are
	// specified, they are considered a single composite key. Structs
	// and
	// repeated data types are not supported; however, nested fields
	// are
	// supported so long as they are not structs themselves or nested
	// within
	// a repeated field.
	QuasiIds []*GooglePrivacyDlpV2beta1FieldId `json:"quasiIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EntityId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EntityId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1KAnonymityConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1KAnonymityConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1KAnonymityEquivalenceClass: The set of
// columns' values that share the same ldiversity value
type GooglePrivacyDlpV2beta1KAnonymityEquivalenceClass struct {
	// EquivalenceClassSize: Size of the equivalence class, for example
	// number of rows with the
	// above set of values.
	EquivalenceClassSize int64 `json:"equivalenceClassSize,omitempty,string"`

	// QuasiIdsValues: Set of values defining the equivalence class. One
	// value per
	// quasi-identifier column in the original KAnonymity metric
	// message.
	// The order is always the same as the original request.
	QuasiIdsValues []*GooglePrivacyDlpV2beta1Value `json:"quasiIdsValues,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "EquivalenceClassSize") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EquivalenceClassSize") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1KAnonymityEquivalenceClass) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1KAnonymityEquivalenceClass
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GooglePrivacyDlpV2beta1KAnonymityHistogramBucket struct {
	// BucketSize: Total number of records in this bucket.
	BucketSize int64 `json:"bucketSize,omitempty,string"`

	// BucketValues: Sample of equivalence classes in this bucket. The total
	// number of
	// classes returned per bucket is capped at 20.
	BucketValues []*GooglePrivacyDlpV2beta1KAnonymityEquivalenceClass `json:"bucketValues,omitempty"`

	// EquivalenceClassSizeLowerBound: Lower bound on the size of the
	// equivalence classes in this bucket.
	EquivalenceClassSizeLowerBound int64 `json:"equivalenceClassSizeLowerBound,omitempty,string"`

	// EquivalenceClassSizeUpperBound: Upper bound on the size of the
	// equivalence classes in this bucket.
	EquivalenceClassSizeUpperBound int64 `json:"equivalenceClassSizeUpperBound,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "BucketSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketSize") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1KAnonymityHistogramBucket) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1KAnonymityHistogramBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1KAnonymityResult: Result of the k-anonymity
// computation.
type GooglePrivacyDlpV2beta1KAnonymityResult struct {
	// EquivalenceClassHistogramBuckets: Histogram of k-anonymity
	// equivalence classes.
	EquivalenceClassHistogramBuckets []*GooglePrivacyDlpV2beta1KAnonymityHistogramBucket `json:"equivalenceClassHistogramBuckets,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "EquivalenceClassHistogramBuckets") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "EquivalenceClassHistogramBuckets") to include in API requests with
	// the JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1KAnonymityResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1KAnonymityResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Key: A unique identifier for a Datastore
// entity.
// If a key's partition ID or any of its path kinds or names
// are
// reserved/read-only, the key is reserved/read-only.
// A reserved/read-only key is forbidden in certain documented contexts.
type GooglePrivacyDlpV2beta1Key struct {
	// PartitionId: Entities are partitioned into subsets, currently
	// identified by a project
	// ID and namespace ID.
	// Queries are scoped to a single partition.
	PartitionId *GooglePrivacyDlpV2beta1PartitionId `json:"partitionId,omitempty"`

	// Path: The entity path.
	// An entity path consists of one or more elements composed of a kind
	// and a
	// string or numerical identifier, which identify entities. The
	// first
	// element identifies a _root entity_, the second element identifies
	// a _child_ of the root entity, the third element identifies a child of
	// the
	// second entity, and so forth. The entities identified by all prefixes
	// of
	// the path are called the element's _ancestors_.
	//
	// A path can never be empty, and a path can have at most 100 elements.
	Path []*GooglePrivacyDlpV2beta1PathElement `json:"path,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PartitionId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PartitionId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Key) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Key
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1KindExpression: A representation of a
// Datastore kind.
type GooglePrivacyDlpV2beta1KindExpression struct {
	// Name: The name of the kind.
	Name string `json:"name,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1KindExpression) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1KindExpression
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1KmsWrappedCryptoKey: Include to use an
// existing data crypto key wrapped by KMS.
// Authorization requires the following IAM permissions when sending a
// request
// to perform a crypto transformation using a kms-wrapped crypto
// key:
// dlp.kms.encrypt
type GooglePrivacyDlpV2beta1KmsWrappedCryptoKey struct {
	// CryptoKeyName: The resource name of the KMS CryptoKey to use for
	// unwrapping. [required]
	CryptoKeyName string `json:"cryptoKeyName,omitempty"`

	// WrappedKey: The wrapped data crypto key. [required]
	WrappedKey string `json:"wrappedKey,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CryptoKeyName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CryptoKeyName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1KmsWrappedCryptoKey) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1KmsWrappedCryptoKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1LDiversityConfig: l-diversity metric, used for
// analysis of reidentification risk.
type GooglePrivacyDlpV2beta1LDiversityConfig struct {
	// QuasiIds: Set of quasi-identifiers indicating how equivalence classes
	// are
	// defined for the l-diversity computation. When multiple fields
	// are
	// specified, they are considered a single composite key.
	QuasiIds []*GooglePrivacyDlpV2beta1FieldId `json:"quasiIds,omitempty"`

	// SensitiveAttribute: Sensitive field for computing the l-value.
	SensitiveAttribute *GooglePrivacyDlpV2beta1FieldId `json:"sensitiveAttribute,omitempty"`

	// ForceSendFields is a list of field names (e.g. "QuasiIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "QuasiIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1LDiversityConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1LDiversityConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1LDiversityEquivalenceClass: The set of
// columns' values that share the same ldiversity value.
type GooglePrivacyDlpV2beta1LDiversityEquivalenceClass struct {
	// EquivalenceClassSize: Size of the k-anonymity equivalence class.
	EquivalenceClassSize int64 `json:"equivalenceClassSize,omitempty,string"`

	// NumDistinctSensitiveValues: Number of distinct sensitive values in
	// this equivalence class.
	NumDistinctSensitiveValues int64 `json:"numDistinctSensitiveValues,omitempty,string"`

	// QuasiIdsValues: Quasi-identifier values defining the k-anonymity
	// equivalence
	// class. The order is always the same as the original request.
	QuasiIdsValues []*GooglePrivacyDlpV2beta1Value `json:"quasiIdsValues,omitempty"`

	// TopSensitiveValues: Estimated frequencies of top sensitive values.
	TopSensitiveValues []*GooglePrivacyDlpV2beta1ValueFrequency `json:"topSensitiveValues,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "EquivalenceClassSize") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EquivalenceClassSize") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1LDiversityEquivalenceClass) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1LDiversityEquivalenceClass
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GooglePrivacyDlpV2beta1LDiversityHistogramBucket struct {
	// BucketSize: Total number of records in this bucket.
	BucketSize int64 `json:"bucketSize,omitempty,string"`

	// BucketValues: Sample of equivalence classes in this bucket. The total
	// number of
	// classes returned per bucket is capped at 20.
	BucketValues []*GooglePrivacyDlpV2beta1LDiversityEquivalenceClass `json:"bucketValues,omitempty"`

	// SensitiveValueFrequencyLowerBound: Lower bound on the sensitive value
	// frequencies of the equivalence
	// classes in this bucket.
	SensitiveValueFrequencyLowerBound int64 `json:"sensitiveValueFrequencyLowerBound,omitempty,string"`

	// SensitiveValueFrequencyUpperBound: Upper bound on the sensitive value
	// frequencies of the equivalence
	// classes in this bucket.
	SensitiveValueFrequencyUpperBound int64 `json:"sensitiveValueFrequencyUpperBound,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "BucketSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketSize") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1LDiversityHistogramBucket) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1LDiversityHistogramBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1LDiversityResult: Result of the l-diversity
// computation.
type GooglePrivacyDlpV2beta1LDiversityResult struct {
	// SensitiveValueFrequencyHistogramBuckets: Histogram of l-diversity
	// equivalence class sensitive value frequencies.
	SensitiveValueFrequencyHistogramBuckets []*GooglePrivacyDlpV2beta1LDiversityHistogramBucket `json:"sensitiveValueFrequencyHistogramBuckets,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "SensitiveValueFrequencyHistogramBuckets") to unconditionally include
	// in API requests. By default, fields with empty values are omitted
	// from API requests. However, any non-pointer, non-interface field
	// appearing in ForceSendFields will be sent to the server regardless of
	// whether the field is empty or not. This may be used to include empty
	// fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "SensitiveValueFrequencyHistogramBuckets") to include in API requests
	// with the JSON null value. By default, fields with empty values are
	// omitted from API requests. However, any field with an empty value
	// appearing in NullFields will be sent to the server as null. It is an
	// error if a field in this list has a non-empty value. This may be used
	// to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1LDiversityResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1LDiversityResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1ListInfoTypesResponse: Response to the
// ListInfoTypes request.
type GooglePrivacyDlpV2beta1ListInfoTypesResponse struct {
	// InfoTypes: Set of sensitive info types belonging to a category.
	InfoTypes []*GooglePrivacyDlpV2beta1InfoTypeDescription `json:"infoTypes,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "InfoTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InfoTypes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ListInfoTypesResponse) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ListInfoTypesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1ListInspectFindingsResponse: Response to the
// ListInspectFindings request.
type GooglePrivacyDlpV2beta1ListInspectFindingsResponse struct {
	// NextPageToken: If not empty, indicates that there may be more results
	// that match the
	// request; this value should be passed in a new
	// `ListInspectFindingsRequest`.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Result: The results.
	Result *GooglePrivacyDlpV2beta1InspectResult `json:"result,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ListInspectFindingsResponse) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ListInspectFindingsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1ListRootCategoriesResponse: Response for
// ListRootCategories request.
type GooglePrivacyDlpV2beta1ListRootCategoriesResponse struct {
	// Categories: List of all into type categories supported by the API.
	Categories []*GooglePrivacyDlpV2beta1CategoryDescription `json:"categories,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Categories") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Categories") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ListRootCategoriesResponse) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ListRootCategoriesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Location: Specifies the location of a finding
// within its source item.
type GooglePrivacyDlpV2beta1Location struct {
	// ByteRange: Zero-based byte offsets within a content item.
	ByteRange *GooglePrivacyDlpV2beta1Range `json:"byteRange,omitempty"`

	// CodepointRange: Character offsets within a content item, included
	// when content type
	// is a text. Default charset assumed to be UTF-8.
	CodepointRange *GooglePrivacyDlpV2beta1Range `json:"codepointRange,omitempty"`

	// FieldId: Field id of the field containing the finding.
	FieldId *GooglePrivacyDlpV2beta1FieldId `json:"fieldId,omitempty"`

	// ImageBoxes: Location within an image's pixels.
	ImageBoxes []*GooglePrivacyDlpV2beta1ImageLocation `json:"imageBoxes,omitempty"`

	// RecordKey: Key of the finding.
	RecordKey *GooglePrivacyDlpV2beta1RecordKey `json:"recordKey,omitempty"`

	// TableLocation: Location within a `ContentItem.Table`.
	TableLocation *GooglePrivacyDlpV2beta1TableLocation `json:"tableLocation,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ByteRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ByteRange") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Location) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Location
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1NumericalStatsConfig: Compute numerical stats
// over an individual column, including
// min, max, and quantiles.
type GooglePrivacyDlpV2beta1NumericalStatsConfig struct {
	// Field: Field to compute numerical stats on. Supported types
	// are
	// integer, float, date, datetime, timestamp, time.
	Field *GooglePrivacyDlpV2beta1FieldId `json:"field,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1NumericalStatsConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1NumericalStatsConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1NumericalStatsResult: Result of the numerical
// stats computation.
type GooglePrivacyDlpV2beta1NumericalStatsResult struct {
	// MaxValue: Maximum value appearing in the column.
	MaxValue *GooglePrivacyDlpV2beta1Value `json:"maxValue,omitempty"`

	// MinValue: Minimum value appearing in the column.
	MinValue *GooglePrivacyDlpV2beta1Value `json:"minValue,omitempty"`

	// QuantileValues: List of 99 values that partition the set of field
	// values into 100 equal
	// sized buckets.
	QuantileValues []*GooglePrivacyDlpV2beta1Value `json:"quantileValues,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxValue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1NumericalStatsResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1NumericalStatsResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1OperationConfig: Additional configuration for
// inspect long running operations.
type GooglePrivacyDlpV2beta1OperationConfig struct {
	// MaxItemFindings: Max number of findings per file, Datastore entity,
	// or database row.
	MaxItemFindings int64 `json:"maxItemFindings,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "MaxItemFindings") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxItemFindings") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1OperationConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1OperationConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1OutputStorageConfig: Cloud repository for
// storing output.
type GooglePrivacyDlpV2beta1OutputStorageConfig struct {
	// StoragePath: The path to a Google Cloud Storage location to store
	// output.
	StoragePath *GooglePrivacyDlpV2beta1CloudStoragePath `json:"storagePath,omitempty"`

	// Table: Store findings in a new table in the dataset.
	Table *GooglePrivacyDlpV2beta1BigQueryTable `json:"table,omitempty"`

	// ForceSendFields is a list of field names (e.g. "StoragePath") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "StoragePath") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1OutputStorageConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1OutputStorageConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1PartitionId: Datastore partition ID.
// A partition ID identifies a grouping of entities. The grouping is
// always
// by project and namespace, however the namespace ID may be empty.
//
// A partition ID contains several dimensions:
// project ID and namespace ID.
type GooglePrivacyDlpV2beta1PartitionId struct {
	// NamespaceId: If not empty, the ID of the namespace to which the
	// entities belong.
	NamespaceId string `json:"namespaceId,omitempty"`

	// ProjectId: The ID of the project to which the entities belong.
	ProjectId string `json:"projectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NamespaceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NamespaceId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1PartitionId) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1PartitionId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1PathElement: A (kind, ID/name) pair used to
// construct a key path.
//
// If either name or ID is set, the element is complete.
// If neither is set, the element is incomplete.
type GooglePrivacyDlpV2beta1PathElement struct {
	// Id: The auto-allocated ID of the entity.
	// Never equal to zero. Values less than zero are discouraged and may
	// not
	// be supported in the future.
	Id int64 `json:"id,omitempty,string"`

	// Kind: The kind of the entity.
	// A kind matching regex `__.*__` is reserved/read-only.
	// A kind must not contain more than 1500 bytes when UTF-8
	// encoded.
	// Cannot be "".
	Kind string `json:"kind,omitempty"`

	// Name: The name of the entity.
	// A name matching regex `__.*__` is reserved/read-only.
	// A name must not be more than 1500 bytes when UTF-8 encoded.
	// Cannot be "".
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1PathElement) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1PathElement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1PrimitiveTransformation: A rule for
// transforming a value.
type GooglePrivacyDlpV2beta1PrimitiveTransformation struct {
	BucketingConfig *GooglePrivacyDlpV2beta1BucketingConfig `json:"bucketingConfig,omitempty"`

	CharacterMaskConfig *GooglePrivacyDlpV2beta1CharacterMaskConfig `json:"characterMaskConfig,omitempty"`

	CryptoHashConfig *GooglePrivacyDlpV2beta1CryptoHashConfig `json:"cryptoHashConfig,omitempty"`

	CryptoReplaceFfxFpeConfig *GooglePrivacyDlpV2beta1CryptoReplaceFfxFpeConfig `json:"cryptoReplaceFfxFpeConfig,omitempty"`

	FixedSizeBucketingConfig *GooglePrivacyDlpV2beta1FixedSizeBucketingConfig `json:"fixedSizeBucketingConfig,omitempty"`

	RedactConfig *GooglePrivacyDlpV2beta1RedactConfig `json:"redactConfig,omitempty"`

	ReplaceConfig *GooglePrivacyDlpV2beta1ReplaceValueConfig `json:"replaceConfig,omitempty"`

	ReplaceWithInfoTypeConfig *GooglePrivacyDlpV2beta1ReplaceWithInfoTypeConfig `json:"replaceWithInfoTypeConfig,omitempty"`

	TimePartConfig *GooglePrivacyDlpV2beta1TimePartConfig `json:"timePartConfig,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BucketingConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketingConfig") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1PrimitiveTransformation) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1PrimitiveTransformation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1PrivacyMetric: Privacy metric to compute for
// reidentification risk analysis.
type GooglePrivacyDlpV2beta1PrivacyMetric struct {
	CategoricalStatsConfig *GooglePrivacyDlpV2beta1CategoricalStatsConfig `json:"categoricalStatsConfig,omitempty"`

	KAnonymityConfig *GooglePrivacyDlpV2beta1KAnonymityConfig `json:"kAnonymityConfig,omitempty"`

	LDiversityConfig *GooglePrivacyDlpV2beta1LDiversityConfig `json:"lDiversityConfig,omitempty"`

	NumericalStatsConfig *GooglePrivacyDlpV2beta1NumericalStatsConfig `json:"numericalStatsConfig,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "CategoricalStatsConfig") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CategoricalStatsConfig")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1PrivacyMetric) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1PrivacyMetric
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Projection: A representation of a Datastore
// property in a projection.
type GooglePrivacyDlpV2beta1Projection struct {
	// Property: The property to project.
	Property *GooglePrivacyDlpV2beta1PropertyReference `json:"property,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Property") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Property") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Projection) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Projection
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1PropertyReference: A reference to a property
// relative to the Datastore kind expressions.
type GooglePrivacyDlpV2beta1PropertyReference struct {
	// Name: The name of the property.
	// If name includes "."s, it may be interpreted as a property name path.
	Name string `json:"name,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1PropertyReference) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1PropertyReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Range: Generic half-open interval [start, end)
type GooglePrivacyDlpV2beta1Range struct {
	// End: Index of the last character of the range (exclusive).
	End int64 `json:"end,omitempty,string"`

	// Start: Index of the first character of the range (inclusive).
	Start int64 `json:"start,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "End") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "End") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Range) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Range
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1RecordCondition: A condition for determing
// whether a transformation should be applied to
// a field.
type GooglePrivacyDlpV2beta1RecordCondition struct {
	Expressions *GooglePrivacyDlpV2beta1Expressions `json:"expressions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Expressions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Expressions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RecordCondition) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RecordCondition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1RecordKey: Message for a unique key indicating
// a record that contains a finding.
type GooglePrivacyDlpV2beta1RecordKey struct {
	CloudStorageKey *GooglePrivacyDlpV2beta1CloudStorageKey `json:"cloudStorageKey,omitempty"`

	DatastoreKey *GooglePrivacyDlpV2beta1DatastoreKey `json:"datastoreKey,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CloudStorageKey") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CloudStorageKey") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RecordKey) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RecordKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1RecordSuppression: Configuration to suppress
// records whose suppression conditions evaluate to
// true.
type GooglePrivacyDlpV2beta1RecordSuppression struct {
	Condition *GooglePrivacyDlpV2beta1RecordCondition `json:"condition,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Condition") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Condition") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RecordSuppression) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RecordSuppression
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1RecordTransformations: A type of
// transformation that is applied over structured data such as a
// table.
type GooglePrivacyDlpV2beta1RecordTransformations struct {
	// FieldTransformations: Transform the record by applying various field
	// transformations.
	FieldTransformations []*GooglePrivacyDlpV2beta1FieldTransformation `json:"fieldTransformations,omitempty"`

	// RecordSuppressions: Configuration defining which records get
	// suppressed entirely. Records that
	// match any suppression rule are omitted from the output [optional].
	RecordSuppressions []*GooglePrivacyDlpV2beta1RecordSuppression `json:"recordSuppressions,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "FieldTransformations") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FieldTransformations") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RecordTransformations) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RecordTransformations
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1RedactConfig: Redact a given value. For
// example, if used with an `InfoTypeTransformation`
// transforming PHONE_NUMBER, and input 'My phone number is
// 206-555-0123', the
// output would be 'My phone number is '.
type GooglePrivacyDlpV2beta1RedactConfig struct {
}

// GooglePrivacyDlpV2beta1RedactContentRequest: Request to search for
// potentially sensitive info in a list of items
// and replace it with a default or provided content.
type GooglePrivacyDlpV2beta1RedactContentRequest struct {
	// ImageRedactionConfigs: The configuration for specifying what content
	// to redact from images.
	ImageRedactionConfigs []*GooglePrivacyDlpV2beta1ImageRedactionConfig `json:"imageRedactionConfigs,omitempty"`

	// InspectConfig: Configuration for the inspector.
	InspectConfig *GooglePrivacyDlpV2beta1InspectConfig `json:"inspectConfig,omitempty"`

	// Items: The list of items to inspect. Up to 100 are allowed per
	// request.
	Items []*GooglePrivacyDlpV2beta1ContentItem `json:"items,omitempty"`

	// ReplaceConfigs: The strings to replace findings text findings with.
	// Must specify at least
	// one of these or one ImageRedactionConfig if redacting images.
	ReplaceConfigs []*GooglePrivacyDlpV2beta1ReplaceConfig `json:"replaceConfigs,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ImageRedactionConfigs") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ImageRedactionConfigs") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RedactContentRequest) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RedactContentRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1RedactContentResponse: Results of redacting a
// list of items.
type GooglePrivacyDlpV2beta1RedactContentResponse struct {
	// Items: The redacted content.
	Items []*GooglePrivacyDlpV2beta1ContentItem `json:"items,omitempty"`

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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RedactContentResponse) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RedactContentResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GooglePrivacyDlpV2beta1ReplaceConfig struct {
	// InfoType: Type of information to replace. Only one ReplaceConfig per
	// info_type
	// should be provided. If ReplaceConfig does not have an info_type, the
	// DLP
	// API matches it against all info_types that are found but not
	// specified in
	// another ReplaceConfig.
	InfoType *GooglePrivacyDlpV2beta1InfoType `json:"infoType,omitempty"`

	// ReplaceWith: Content replacing sensitive information of given type.
	// Max 256 chars.
	ReplaceWith string `json:"replaceWith,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InfoType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InfoType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ReplaceConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ReplaceConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1ReplaceValueConfig: Replace each input value
// with a given `Value`.
type GooglePrivacyDlpV2beta1ReplaceValueConfig struct {
	// NewValue: Value to replace it with.
	NewValue *GooglePrivacyDlpV2beta1Value `json:"newValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NewValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NewValue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1ReplaceValueConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ReplaceValueConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1ReplaceWithInfoTypeConfig: Replace each
// matching finding with the name of the info_type.
type GooglePrivacyDlpV2beta1ReplaceWithInfoTypeConfig struct {
}

// GooglePrivacyDlpV2beta1RiskAnalysisOperationMetadata: Metadata
// returned within GetOperation for risk analysis.
type GooglePrivacyDlpV2beta1RiskAnalysisOperationMetadata struct {
	// CreateTime: The time which this request was started.
	CreateTime string `json:"createTime,omitempty"`

	// RequestedPrivacyMetric: Privacy metric to compute.
	RequestedPrivacyMetric *GooglePrivacyDlpV2beta1PrivacyMetric `json:"requestedPrivacyMetric,omitempty"`

	// RequestedSourceTable: Input dataset to compute metrics over.
	RequestedSourceTable *GooglePrivacyDlpV2beta1BigQueryTable `json:"requestedSourceTable,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RiskAnalysisOperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RiskAnalysisOperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1RiskAnalysisOperationResult: Result of a risk
// analysis operation request.
type GooglePrivacyDlpV2beta1RiskAnalysisOperationResult struct {
	CategoricalStatsResult *GooglePrivacyDlpV2beta1CategoricalStatsResult `json:"categoricalStatsResult,omitempty"`

	KAnonymityResult *GooglePrivacyDlpV2beta1KAnonymityResult `json:"kAnonymityResult,omitempty"`

	LDiversityResult *GooglePrivacyDlpV2beta1LDiversityResult `json:"lDiversityResult,omitempty"`

	NumericalStatsResult *GooglePrivacyDlpV2beta1NumericalStatsResult `json:"numericalStatsResult,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "CategoricalStatsResult") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CategoricalStatsResult")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1RiskAnalysisOperationResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1RiskAnalysisOperationResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GooglePrivacyDlpV2beta1Row struct {
	Values []*GooglePrivacyDlpV2beta1Value `json:"values,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1Row) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Row
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1StorageConfig: Shared message indicating Cloud
// storage type.
type GooglePrivacyDlpV2beta1StorageConfig struct {
	// BigQueryOptions: BigQuery options specification.
	BigQueryOptions *GooglePrivacyDlpV2beta1BigQueryOptions `json:"bigQueryOptions,omitempty"`

	// CloudStorageOptions: Google Cloud Storage options specification.
	CloudStorageOptions *GooglePrivacyDlpV2beta1CloudStorageOptions `json:"cloudStorageOptions,omitempty"`

	// DatastoreOptions: Google Cloud Datastore options specification.
	DatastoreOptions *GooglePrivacyDlpV2beta1DatastoreOptions `json:"datastoreOptions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BigQueryOptions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BigQueryOptions") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1StorageConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1StorageConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1SummaryResult: A collection that informs the
// user the number of times a particular
// `TransformationResultCode` and error details occurred.
type GooglePrivacyDlpV2beta1SummaryResult struct {
	// Possible values:
	//   "TRANSFORMATION_RESULT_CODE_UNSPECIFIED"
	//   "SUCCESS"
	//   "ERROR"
	Code string `json:"code,omitempty"`

	Count int64 `json:"count,omitempty,string"`

	// Details: A place for warnings or errors to show up if a
	// transformation didn't
	// work as expected.
	Details string `json:"details,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1SummaryResult) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1SummaryResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Table: Structured content to inspect. Up to
// 50,000 `Value`s per request allowed.
type GooglePrivacyDlpV2beta1Table struct {
	Headers []*GooglePrivacyDlpV2beta1FieldId `json:"headers,omitempty"`

	Rows []*GooglePrivacyDlpV2beta1Row `json:"rows,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Headers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Headers") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Table) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Table
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1TableLocation: Location of a finding within a
// `ContentItem.Table`.
type GooglePrivacyDlpV2beta1TableLocation struct {
	// RowIndex: The zero-based index of the row where the finding is
	// located.
	RowIndex int64 `json:"rowIndex,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "RowIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "RowIndex") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1TableLocation) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1TableLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1TimePartConfig: For use with `Date`,
// `Timestamp`, and `TimeOfDay`, extract or preserve a
// portion of the value.
type GooglePrivacyDlpV2beta1TimePartConfig struct {
	// Possible values:
	//   "TIME_PART_UNSPECIFIED"
	//   "YEAR" - [000-9999]
	//   "MONTH" - [1-12]
	//   "DAY_OF_MONTH" - [1-31]
	//   "DAY_OF_WEEK" - [1-7]
	//   "WEEK_OF_YEAR" - [1-52]
	//   "HOUR_OF_DAY" - [0-24]
	PartToExtract string `json:"partToExtract,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PartToExtract") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PartToExtract") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1TimePartConfig) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1TimePartConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1TransformationSummary: Summary of a single
// tranformation.
type GooglePrivacyDlpV2beta1TransformationSummary struct {
	// Field: Set if the transformation was limited to a specific FieldId.
	Field *GooglePrivacyDlpV2beta1FieldId `json:"field,omitempty"`

	// FieldTransformations: The field transformation that was applied. This
	// list will contain
	// multiple only in the case of errors.
	FieldTransformations []*GooglePrivacyDlpV2beta1FieldTransformation `json:"fieldTransformations,omitempty"`

	// InfoType: Set if the transformation was limited to a specific
	// info_type.
	InfoType *GooglePrivacyDlpV2beta1InfoType `json:"infoType,omitempty"`

	// RecordSuppress: The specific suppression option these stats apply to.
	RecordSuppress *GooglePrivacyDlpV2beta1RecordSuppression `json:"recordSuppress,omitempty"`

	Results []*GooglePrivacyDlpV2beta1SummaryResult `json:"results,omitempty"`

	// Transformation: The specific transformation these stats apply to.
	Transformation *GooglePrivacyDlpV2beta1PrimitiveTransformation `json:"transformation,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1TransformationSummary) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1TransformationSummary
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1TransientCryptoKey: Use this to have a random
// data crypto key generated.
// It will be discarded after the operation/request finishes.
type GooglePrivacyDlpV2beta1TransientCryptoKey struct {
	// Name: Name of the key. [required]
	// This is an arbitrary string used to differentiate different keys.
	// A unique key is generated per name: two separate
	// `TransientCryptoKey`
	// protos share the same generated key if their names are the same.
	// When the data crypto key is generated, this name is not used in any
	// way
	// (repeating the api call will result in a different key being
	// generated).
	Name string `json:"name,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1TransientCryptoKey) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1TransientCryptoKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1UnwrappedCryptoKey: Using raw keys is prone to
// security risks due to accidentally
// leaking the key. Choose another type of key if possible.
type GooglePrivacyDlpV2beta1UnwrappedCryptoKey struct {
	// Key: The AES 128/192/256 bit key. [required]
	Key string `json:"key,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1UnwrappedCryptoKey) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1UnwrappedCryptoKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GooglePrivacyDlpV2beta1Value: Set of primitive values supported by
// the system.
type GooglePrivacyDlpV2beta1Value struct {
	BooleanValue bool `json:"booleanValue,omitempty"`

	DateValue *GoogleTypeDate `json:"dateValue,omitempty"`

	FloatValue float64 `json:"floatValue,omitempty"`

	IntegerValue int64 `json:"integerValue,omitempty,string"`

	StringValue string `json:"stringValue,omitempty"`

	TimeValue *GoogleTypeTimeOfDay `json:"timeValue,omitempty"`

	TimestampValue string `json:"timestampValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BooleanValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BooleanValue") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GooglePrivacyDlpV2beta1Value) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1Value
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *GooglePrivacyDlpV2beta1Value) UnmarshalJSON(data []byte) error {
	type noMethod GooglePrivacyDlpV2beta1Value
	var s1 struct {
		FloatValue gensupport.JSONFloat64 `json:"floatValue"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.FloatValue = float64(s1.FloatValue)
	return nil
}

// GooglePrivacyDlpV2beta1ValueFrequency: A value of a field, including
// its frequency.
type GooglePrivacyDlpV2beta1ValueFrequency struct {
	// Count: How many times the value is contained in the field.
	Count int64 `json:"count,omitempty,string"`

	// Value: A value contained in the field in question.
	Value *GooglePrivacyDlpV2beta1Value `json:"value,omitempty"`

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

func (s *GooglePrivacyDlpV2beta1ValueFrequency) MarshalJSON() ([]byte, error) {
	type noMethod GooglePrivacyDlpV2beta1ValueFrequency
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleProtobufEmpty: A generic empty message that you can re-use to
// avoid defining duplicated
// empty messages in your APIs. A typical example is to use it as the
// request
// or the response type of an API method. For instance:
//
//     service Foo {
//       rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
//     }
//
// The JSON representation for `Empty` is empty JSON object `{}`.
type GoogleProtobufEmpty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// GoogleRpcStatus: The `Status` type defines a logical error model that
// is suitable for different
// programming environments, including REST APIs and RPC APIs. It is
// used by
// [gRPC](https://github.com/grpc). The error model is designed to
// be:
//
// - Simple to use and understand for most users
// - Flexible enough to meet unexpected needs
//
// # Overview
//
// The `Status` message contains three pieces of data: error code, error
// message,
// and error details. The error code should be an enum value
// of
// google.rpc.Code, but it may accept additional error codes if needed.
// The
// error message should be a developer-facing English message that
// helps
// developers *understand* and *resolve* the error. If a localized
// user-facing
// error message is needed, put the localized message in the error
// details or
// localize it in the client. The optional error details may contain
// arbitrary
// information about the error. There is a predefined set of error
// detail types
// in the package `google.rpc` that can be used for common error
// conditions.
//
// # Language mapping
//
// The `Status` message is the logical representation of the error
// model, but it
// is not necessarily the actual wire format. When the `Status` message
// is
// exposed in different client libraries and different wire protocols,
// it can be
// mapped differently. For example, it will likely be mapped to some
// exceptions
// in Java, but more likely mapped to some error codes in C.
//
// # Other uses
//
// The error model and the `Status` message can be used in a variety
// of
// environments, either with or without APIs, to provide a
// consistent developer experience across different
// environments.
//
// Example uses of this error model include:
//
// - Partial errors. If a service needs to return partial errors to the
// client,
//     it may embed the `Status` in the normal response to indicate the
// partial
//     errors.
//
// - Workflow errors. A typical workflow has multiple steps. Each step
// may
//     have a `Status` message for error reporting.
//
// - Batch operations. If a client uses batch request and batch
// response, the
//     `Status` message should be used directly inside batch response,
// one for
//     each error sub-response.
//
// - Asynchronous operations. If an API call embeds asynchronous
// operation
//     results in its response, the status of those operations should
// be
//     represented directly using the `Status` message.
//
// - Logging. If some API errors are stored in logs, the message
// `Status` could
//     be used directly after any stripping needed for security/privacy
// reasons.
type GoogleRpcStatus struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details.  There is a
	// common set of
	// message types for APIs to use.
	Details []googleapi.RawMessage `json:"details,omitempty"`

	// Message: A developer-facing error message, which should be in
	// English. Any
	// user-facing error message should be localized and sent in
	// the
	// google.rpc.Status.details field, or localized by the client.
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

func (s *GoogleRpcStatus) MarshalJSON() ([]byte, error) {
	type noMethod GoogleRpcStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleTypeDate: Represents a whole calendar date, e.g. date of birth.
// The time of day and
// time zone are either specified elsewhere or are not significant. The
// date
// is relative to the Proleptic Gregorian Calendar. The day may be 0
// to
// represent a year and month where the day is not significant, e.g.
// credit card
// expiration date. The year may be 0 to represent a month and day
// independent
// of year, e.g. anniversary date. Related types are
// google.type.TimeOfDay
// and `google.protobuf.Timestamp`.
type GoogleTypeDate struct {
	// Day: Day of month. Must be from 1 to 31 and valid for the year and
	// month, or 0
	// if specifying a year/month where the day is not significant.
	Day int64 `json:"day,omitempty"`

	// Month: Month of year. Must be from 1 to 12.
	Month int64 `json:"month,omitempty"`

	// Year: Year of date. Must be from 1 to 9999, or 0 if specifying a date
	// without
	// a year.
	Year int64 `json:"year,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Day") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Day") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleTypeDate) MarshalJSON() ([]byte, error) {
	type noMethod GoogleTypeDate
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleTypeTimeOfDay: Represents a time of day. The date and time zone
// are either not significant
// or are specified elsewhere. An API may choose to allow leap seconds.
// Related
// types are google.type.Date and `google.protobuf.Timestamp`.
type GoogleTypeTimeOfDay struct {
	// Hours: Hours of day in 24 hour format. Should be from 0 to 23. An API
	// may choose
	// to allow the value "24:00:00" for scenarios like business closing
	// time.
	Hours int64 `json:"hours,omitempty"`

	// Minutes: Minutes of hour of day. Must be from 0 to 59.
	Minutes int64 `json:"minutes,omitempty"`

	// Nanos: Fractions of seconds in nanoseconds. Must be from 0 to
	// 999,999,999.
	Nanos int64 `json:"nanos,omitempty"`

	// Seconds: Seconds of minutes of the time. Must normally be from 0 to
	// 59. An API may
	// allow the value 60 if it allows leap-seconds.
	Seconds int64 `json:"seconds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Hours") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Hours") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleTypeTimeOfDay) MarshalJSON() ([]byte, error) {
	type noMethod GoogleTypeTimeOfDay
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "dlp.content.deidentify":

type ContentDeidentifyCall struct {
	s                                               *Service
	googleprivacydlpv2beta1deidentifycontentrequest *GooglePrivacyDlpV2beta1DeidentifyContentRequest
	urlParams_                                      gensupport.URLParams
	ctx_                                            context.Context
	header_                                         http.Header
}

// Deidentify: De-identifies potentially sensitive info from a list of
// strings.
// This method has limits on input size and output size.
func (r *ContentService) Deidentify(googleprivacydlpv2beta1deidentifycontentrequest *GooglePrivacyDlpV2beta1DeidentifyContentRequest) *ContentDeidentifyCall {
	c := &ContentDeidentifyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.googleprivacydlpv2beta1deidentifycontentrequest = googleprivacydlpv2beta1deidentifycontentrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContentDeidentifyCall) Fields(s ...googleapi.Field) *ContentDeidentifyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContentDeidentifyCall) Context(ctx context.Context) *ContentDeidentifyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContentDeidentifyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContentDeidentifyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googleprivacydlpv2beta1deidentifycontentrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/content:deidentify")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.content.deidentify" call.
// Exactly one of *GooglePrivacyDlpV2beta1DeidentifyContentResponse or
// error will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *GooglePrivacyDlpV2beta1DeidentifyContentResponse.ServerResponse.Heade
// r or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ContentDeidentifyCall) Do(opts ...googleapi.CallOption) (*GooglePrivacyDlpV2beta1DeidentifyContentResponse, error) {
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
	ret := &GooglePrivacyDlpV2beta1DeidentifyContentResponse{
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
	//   "description": "De-identifies potentially sensitive info from a list of strings.\nThis method has limits on input size and output size.",
	//   "flatPath": "v2beta1/content:deidentify",
	//   "httpMethod": "POST",
	//   "id": "dlp.content.deidentify",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/content:deidentify",
	//   "request": {
	//     "$ref": "GooglePrivacyDlpV2beta1DeidentifyContentRequest"
	//   },
	//   "response": {
	//     "$ref": "GooglePrivacyDlpV2beta1DeidentifyContentResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.content.inspect":

type ContentInspectCall struct {
	s                                            *Service
	googleprivacydlpv2beta1inspectcontentrequest *GooglePrivacyDlpV2beta1InspectContentRequest
	urlParams_                                   gensupport.URLParams
	ctx_                                         context.Context
	header_                                      http.Header
}

// Inspect: Finds potentially sensitive info in a list of strings.
// This method has limits on input size, processing time, and output
// size.
func (r *ContentService) Inspect(googleprivacydlpv2beta1inspectcontentrequest *GooglePrivacyDlpV2beta1InspectContentRequest) *ContentInspectCall {
	c := &ContentInspectCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.googleprivacydlpv2beta1inspectcontentrequest = googleprivacydlpv2beta1inspectcontentrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContentInspectCall) Fields(s ...googleapi.Field) *ContentInspectCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContentInspectCall) Context(ctx context.Context) *ContentInspectCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContentInspectCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContentInspectCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googleprivacydlpv2beta1inspectcontentrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/content:inspect")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.content.inspect" call.
// Exactly one of *GooglePrivacyDlpV2beta1InspectContentResponse or
// error will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *GooglePrivacyDlpV2beta1InspectContentResponse.ServerResponse.Header
// or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ContentInspectCall) Do(opts ...googleapi.CallOption) (*GooglePrivacyDlpV2beta1InspectContentResponse, error) {
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
	ret := &GooglePrivacyDlpV2beta1InspectContentResponse{
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
	//   "description": "Finds potentially sensitive info in a list of strings.\nThis method has limits on input size, processing time, and output size.",
	//   "flatPath": "v2beta1/content:inspect",
	//   "httpMethod": "POST",
	//   "id": "dlp.content.inspect",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/content:inspect",
	//   "request": {
	//     "$ref": "GooglePrivacyDlpV2beta1InspectContentRequest"
	//   },
	//   "response": {
	//     "$ref": "GooglePrivacyDlpV2beta1InspectContentResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.content.redact":

type ContentRedactCall struct {
	s                                           *Service
	googleprivacydlpv2beta1redactcontentrequest *GooglePrivacyDlpV2beta1RedactContentRequest
	urlParams_                                  gensupport.URLParams
	ctx_                                        context.Context
	header_                                     http.Header
}

// Redact: Redacts potentially sensitive info from a list of
// strings.
// This method has limits on input size, processing time, and output
// size.
func (r *ContentService) Redact(googleprivacydlpv2beta1redactcontentrequest *GooglePrivacyDlpV2beta1RedactContentRequest) *ContentRedactCall {
	c := &ContentRedactCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.googleprivacydlpv2beta1redactcontentrequest = googleprivacydlpv2beta1redactcontentrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContentRedactCall) Fields(s ...googleapi.Field) *ContentRedactCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContentRedactCall) Context(ctx context.Context) *ContentRedactCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContentRedactCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContentRedactCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googleprivacydlpv2beta1redactcontentrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/content:redact")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.content.redact" call.
// Exactly one of *GooglePrivacyDlpV2beta1RedactContentResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *GooglePrivacyDlpV2beta1RedactContentResponse.ServerResponse.Header
// or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ContentRedactCall) Do(opts ...googleapi.CallOption) (*GooglePrivacyDlpV2beta1RedactContentResponse, error) {
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
	ret := &GooglePrivacyDlpV2beta1RedactContentResponse{
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
	//   "description": "Redacts potentially sensitive info from a list of strings.\nThis method has limits on input size, processing time, and output size.",
	//   "flatPath": "v2beta1/content:redact",
	//   "httpMethod": "POST",
	//   "id": "dlp.content.redact",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/content:redact",
	//   "request": {
	//     "$ref": "GooglePrivacyDlpV2beta1RedactContentRequest"
	//   },
	//   "response": {
	//     "$ref": "GooglePrivacyDlpV2beta1RedactContentResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.dataSource.analyze":

type DataSourceAnalyzeCall struct {
	s                                                   *Service
	googleprivacydlpv2beta1analyzedatasourceriskrequest *GooglePrivacyDlpV2beta1AnalyzeDataSourceRiskRequest
	urlParams_                                          gensupport.URLParams
	ctx_                                                context.Context
	header_                                             http.Header
}

// Analyze: Schedules a job to compute risk analysis metrics over
// content in a Google
// Cloud Platform repository.
func (r *DataSourceService) Analyze(googleprivacydlpv2beta1analyzedatasourceriskrequest *GooglePrivacyDlpV2beta1AnalyzeDataSourceRiskRequest) *DataSourceAnalyzeCall {
	c := &DataSourceAnalyzeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.googleprivacydlpv2beta1analyzedatasourceriskrequest = googleprivacydlpv2beta1analyzedatasourceriskrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DataSourceAnalyzeCall) Fields(s ...googleapi.Field) *DataSourceAnalyzeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DataSourceAnalyzeCall) Context(ctx context.Context) *DataSourceAnalyzeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DataSourceAnalyzeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DataSourceAnalyzeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googleprivacydlpv2beta1analyzedatasourceriskrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/dataSource:analyze")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.dataSource.analyze" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DataSourceAnalyzeCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	ret := &GoogleLongrunningOperation{
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
	//   "description": "Schedules a job to compute risk analysis metrics over content in a Google\nCloud Platform repository.",
	//   "flatPath": "v2beta1/dataSource:analyze",
	//   "httpMethod": "POST",
	//   "id": "dlp.dataSource.analyze",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/dataSource:analyze",
	//   "request": {
	//     "$ref": "GooglePrivacyDlpV2beta1AnalyzeDataSourceRiskRequest"
	//   },
	//   "response": {
	//     "$ref": "GoogleLongrunningOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.inspect.operations.cancel":

type InspectOperationsCancelCall struct {
	s                                       *Service
	name                                    string
	googlelongrunningcanceloperationrequest *GoogleLongrunningCancelOperationRequest
	urlParams_                              gensupport.URLParams
	ctx_                                    context.Context
	header_                                 http.Header
}

// Cancel: Cancels an operation. Use the get method to check whether the
// cancellation succeeded or whether the operation completed despite
// cancellation.
func (r *InspectOperationsService) Cancel(name string, googlelongrunningcanceloperationrequest *GoogleLongrunningCancelOperationRequest) *InspectOperationsCancelCall {
	c := &InspectOperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.googlelongrunningcanceloperationrequest = googlelongrunningcanceloperationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InspectOperationsCancelCall) Fields(s ...googleapi.Field) *InspectOperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InspectOperationsCancelCall) Context(ctx context.Context) *InspectOperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InspectOperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InspectOperationsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googlelongrunningcanceloperationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.inspect.operations.cancel" call.
// Exactly one of *GoogleProtobufEmpty or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GoogleProtobufEmpty.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InspectOperationsCancelCall) Do(opts ...googleapi.CallOption) (*GoogleProtobufEmpty, error) {
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
	ret := &GoogleProtobufEmpty{
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
	//   "description": "Cancels an operation. Use the get method to check whether the cancellation succeeded or whether the operation completed despite cancellation.",
	//   "flatPath": "v2beta1/inspect/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "dlp.inspect.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^inspect/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}:cancel",
	//   "request": {
	//     "$ref": "GoogleLongrunningCancelOperationRequest"
	//   },
	//   "response": {
	//     "$ref": "GoogleProtobufEmpty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.inspect.operations.create":

type InspectOperationsCreateCall struct {
	s                                                    *Service
	googleprivacydlpv2beta1createinspectoperationrequest *GooglePrivacyDlpV2beta1CreateInspectOperationRequest
	urlParams_                                           gensupport.URLParams
	ctx_                                                 context.Context
	header_                                              http.Header
}

// Create: Schedules a job scanning content in a Google Cloud Platform
// data
// repository.
func (r *InspectOperationsService) Create(googleprivacydlpv2beta1createinspectoperationrequest *GooglePrivacyDlpV2beta1CreateInspectOperationRequest) *InspectOperationsCreateCall {
	c := &InspectOperationsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.googleprivacydlpv2beta1createinspectoperationrequest = googleprivacydlpv2beta1createinspectoperationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InspectOperationsCreateCall) Fields(s ...googleapi.Field) *InspectOperationsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InspectOperationsCreateCall) Context(ctx context.Context) *InspectOperationsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InspectOperationsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InspectOperationsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googleprivacydlpv2beta1createinspectoperationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/inspect/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.inspect.operations.create" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InspectOperationsCreateCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	ret := &GoogleLongrunningOperation{
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
	//   "description": "Schedules a job scanning content in a Google Cloud Platform data\nrepository.",
	//   "flatPath": "v2beta1/inspect/operations",
	//   "httpMethod": "POST",
	//   "id": "dlp.inspect.operations.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/inspect/operations",
	//   "request": {
	//     "$ref": "GooglePrivacyDlpV2beta1CreateInspectOperationRequest"
	//   },
	//   "response": {
	//     "$ref": "GoogleLongrunningOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.inspect.operations.delete":

type InspectOperationsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: This method is not supported and the server returns
// `UNIMPLEMENTED`.
func (r *InspectOperationsService) Delete(name string) *InspectOperationsDeleteCall {
	c := &InspectOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InspectOperationsDeleteCall) Fields(s ...googleapi.Field) *InspectOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InspectOperationsDeleteCall) Context(ctx context.Context) *InspectOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InspectOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InspectOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.inspect.operations.delete" call.
// Exactly one of *GoogleProtobufEmpty or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GoogleProtobufEmpty.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InspectOperationsDeleteCall) Do(opts ...googleapi.CallOption) (*GoogleProtobufEmpty, error) {
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
	ret := &GoogleProtobufEmpty{
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
	//   "description": "This method is not supported and the server returns `UNIMPLEMENTED`.",
	//   "flatPath": "v2beta1/inspect/operations/{operationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "dlp.inspect.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^inspect/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "GoogleProtobufEmpty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.inspect.operations.get":

type InspectOperationsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the latest state of a long-running operation.  Clients can
// use this
// method to poll the operation result at intervals as recommended by
// the API
// service.
func (r *InspectOperationsService) Get(name string) *InspectOperationsGetCall {
	c := &InspectOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InspectOperationsGetCall) Fields(s ...googleapi.Field) *InspectOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InspectOperationsGetCall) IfNoneMatch(entityTag string) *InspectOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InspectOperationsGetCall) Context(ctx context.Context) *InspectOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InspectOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InspectOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.inspect.operations.get" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InspectOperationsGetCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	ret := &GoogleLongrunningOperation{
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
	//   "description": "Gets the latest state of a long-running operation.  Clients can use this\nmethod to poll the operation result at intervals as recommended by the API\nservice.",
	//   "flatPath": "v2beta1/inspect/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "dlp.inspect.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^inspect/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "GoogleLongrunningOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.inspect.operations.list":

type InspectOperationsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Fetch the list of long running operations.
func (r *InspectOperationsService) List(name string) *InspectOperationsListCall {
	c := &InspectOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": This parameter supports
// filtering by done, ie done=true or done=false.
func (c *InspectOperationsListCall) Filter(filter string) *InspectOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The list page size.
// The max allowed value is 256 and default is 100.
func (c *InspectOperationsListCall) PageSize(pageSize int64) *InspectOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *InspectOperationsListCall) PageToken(pageToken string) *InspectOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InspectOperationsListCall) Fields(s ...googleapi.Field) *InspectOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InspectOperationsListCall) IfNoneMatch(entityTag string) *InspectOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InspectOperationsListCall) Context(ctx context.Context) *InspectOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InspectOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InspectOperationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.inspect.operations.list" call.
// Exactly one of *GoogleLongrunningListOperationsResponse or error will
// be non-nil. Any non-2xx status code is an error. Response headers are
// in either
// *GoogleLongrunningListOperationsResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InspectOperationsListCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningListOperationsResponse, error) {
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
	ret := &GoogleLongrunningListOperationsResponse{
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
	//   "description": "Fetch the list of long running operations.",
	//   "flatPath": "v2beta1/inspect/operations",
	//   "httpMethod": "GET",
	//   "id": "dlp.inspect.operations.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "This parameter supports filtering by done, ie done=true or done=false.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The name of the operation's parent resource.",
	//       "location": "path",
	//       "pattern": "^inspect/operations$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The list page size. The max allowed value is 256 and default is 100.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The standard list page token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "GoogleLongrunningListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *InspectOperationsListCall) Pages(ctx context.Context, f func(*GoogleLongrunningListOperationsResponse) error) error {
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

// method id "dlp.inspect.results.findings.list":

type InspectResultsFindingsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Returns list of results for given inspect operation result set
// id.
func (r *InspectResultsFindingsService) List(name string) *InspectResultsFindingsListCall {
	c := &InspectResultsFindingsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": Restricts findings to
// items that match. Supports info_type and likelihood.
//
// Examples:
//
// - info_type=EMAIL_ADDRESS
// - info_type=PHONE_NUMBER,EMAIL_ADDRESS
// - likelihood=VERY_LIKELY
// - likelihood=VERY_LIKELY,LIKELY
// - info_type=EMAIL_ADDRESS,likelihood=VERY_LIKELY,LIKELY
func (c *InspectResultsFindingsListCall) Filter(filter string) *InspectResultsFindingsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// results to return.
// If 0, the implementation selects a reasonable value.
func (c *InspectResultsFindingsListCall) PageSize(pageSize int64) *InspectResultsFindingsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The value returned
// by the last `ListInspectFindingsResponse`; indicates
// that this is a continuation of a prior `ListInspectFindings` call,
// and that
// the system should return the next page of data.
func (c *InspectResultsFindingsListCall) PageToken(pageToken string) *InspectResultsFindingsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InspectResultsFindingsListCall) Fields(s ...googleapi.Field) *InspectResultsFindingsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InspectResultsFindingsListCall) IfNoneMatch(entityTag string) *InspectResultsFindingsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InspectResultsFindingsListCall) Context(ctx context.Context) *InspectResultsFindingsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InspectResultsFindingsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InspectResultsFindingsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}/findings")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.inspect.results.findings.list" call.
// Exactly one of *GooglePrivacyDlpV2beta1ListInspectFindingsResponse or
// error will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *GooglePrivacyDlpV2beta1ListInspectFindingsResponse.ServerResponse.Hea
// der or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *InspectResultsFindingsListCall) Do(opts ...googleapi.CallOption) (*GooglePrivacyDlpV2beta1ListInspectFindingsResponse, error) {
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
	ret := &GooglePrivacyDlpV2beta1ListInspectFindingsResponse{
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
	//   "description": "Returns list of results for given inspect operation result set id.",
	//   "flatPath": "v2beta1/inspect/results/{resultsId}/findings",
	//   "httpMethod": "GET",
	//   "id": "dlp.inspect.results.findings.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Restricts findings to items that match. Supports info_type and likelihood.\n\nExamples:\n\n- info_type=EMAIL_ADDRESS\n- info_type=PHONE_NUMBER,EMAIL_ADDRESS\n- likelihood=VERY_LIKELY\n- likelihood=VERY_LIKELY,LIKELY\n- info_type=EMAIL_ADDRESS,likelihood=VERY_LIKELY,LIKELY",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Identifier of the results set returned as metadata of\nthe longrunning operation created by a call to InspectDataSource.\nShould be in the format of `inspect/results/{id}`.",
	//       "location": "path",
	//       "pattern": "^inspect/results/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum number of results to return.\nIf 0, the implementation selects a reasonable value.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value returned by the last `ListInspectFindingsResponse`; indicates\nthat this is a continuation of a prior `ListInspectFindings` call, and that\nthe system should return the next page of data.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}/findings",
	//   "response": {
	//     "$ref": "GooglePrivacyDlpV2beta1ListInspectFindingsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *InspectResultsFindingsListCall) Pages(ctx context.Context, f func(*GooglePrivacyDlpV2beta1ListInspectFindingsResponse) error) error {
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

// method id "dlp.riskAnalysis.operations.cancel":

type RiskAnalysisOperationsCancelCall struct {
	s                                       *Service
	name                                    string
	googlelongrunningcanceloperationrequest *GoogleLongrunningCancelOperationRequest
	urlParams_                              gensupport.URLParams
	ctx_                                    context.Context
	header_                                 http.Header
}

// Cancel: Cancels an operation. Use the get method to check whether the
// cancellation succeeded or whether the operation completed despite
// cancellation.
func (r *RiskAnalysisOperationsService) Cancel(name string, googlelongrunningcanceloperationrequest *GoogleLongrunningCancelOperationRequest) *RiskAnalysisOperationsCancelCall {
	c := &RiskAnalysisOperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.googlelongrunningcanceloperationrequest = googlelongrunningcanceloperationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RiskAnalysisOperationsCancelCall) Fields(s ...googleapi.Field) *RiskAnalysisOperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RiskAnalysisOperationsCancelCall) Context(ctx context.Context) *RiskAnalysisOperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RiskAnalysisOperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RiskAnalysisOperationsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googlelongrunningcanceloperationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.riskAnalysis.operations.cancel" call.
// Exactly one of *GoogleProtobufEmpty or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GoogleProtobufEmpty.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RiskAnalysisOperationsCancelCall) Do(opts ...googleapi.CallOption) (*GoogleProtobufEmpty, error) {
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
	ret := &GoogleProtobufEmpty{
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
	//   "description": "Cancels an operation. Use the get method to check whether the cancellation succeeded or whether the operation completed despite cancellation.",
	//   "flatPath": "v2beta1/riskAnalysis/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "dlp.riskAnalysis.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^riskAnalysis/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}:cancel",
	//   "request": {
	//     "$ref": "GoogleLongrunningCancelOperationRequest"
	//   },
	//   "response": {
	//     "$ref": "GoogleProtobufEmpty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.riskAnalysis.operations.delete":

type RiskAnalysisOperationsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: This method is not supported and the server returns
// `UNIMPLEMENTED`.
func (r *RiskAnalysisOperationsService) Delete(name string) *RiskAnalysisOperationsDeleteCall {
	c := &RiskAnalysisOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RiskAnalysisOperationsDeleteCall) Fields(s ...googleapi.Field) *RiskAnalysisOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RiskAnalysisOperationsDeleteCall) Context(ctx context.Context) *RiskAnalysisOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RiskAnalysisOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RiskAnalysisOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.riskAnalysis.operations.delete" call.
// Exactly one of *GoogleProtobufEmpty or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GoogleProtobufEmpty.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RiskAnalysisOperationsDeleteCall) Do(opts ...googleapi.CallOption) (*GoogleProtobufEmpty, error) {
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
	ret := &GoogleProtobufEmpty{
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
	//   "description": "This method is not supported and the server returns `UNIMPLEMENTED`.",
	//   "flatPath": "v2beta1/riskAnalysis/operations/{operationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "dlp.riskAnalysis.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^riskAnalysis/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "GoogleProtobufEmpty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.riskAnalysis.operations.get":

type RiskAnalysisOperationsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the latest state of a long-running operation.  Clients can
// use this
// method to poll the operation result at intervals as recommended by
// the API
// service.
func (r *RiskAnalysisOperationsService) Get(name string) *RiskAnalysisOperationsGetCall {
	c := &RiskAnalysisOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RiskAnalysisOperationsGetCall) Fields(s ...googleapi.Field) *RiskAnalysisOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RiskAnalysisOperationsGetCall) IfNoneMatch(entityTag string) *RiskAnalysisOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RiskAnalysisOperationsGetCall) Context(ctx context.Context) *RiskAnalysisOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RiskAnalysisOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RiskAnalysisOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.riskAnalysis.operations.get" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RiskAnalysisOperationsGetCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	ret := &GoogleLongrunningOperation{
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
	//   "description": "Gets the latest state of a long-running operation.  Clients can use this\nmethod to poll the operation result at intervals as recommended by the API\nservice.",
	//   "flatPath": "v2beta1/riskAnalysis/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "dlp.riskAnalysis.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^riskAnalysis/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "GoogleLongrunningOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.riskAnalysis.operations.list":

type RiskAnalysisOperationsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Fetch the list of long running operations.
func (r *RiskAnalysisOperationsService) List(name string) *RiskAnalysisOperationsListCall {
	c := &RiskAnalysisOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": This parameter supports
// filtering by done, ie done=true or done=false.
func (c *RiskAnalysisOperationsListCall) Filter(filter string) *RiskAnalysisOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The list page size.
// The max allowed value is 256 and default is 100.
func (c *RiskAnalysisOperationsListCall) PageSize(pageSize int64) *RiskAnalysisOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *RiskAnalysisOperationsListCall) PageToken(pageToken string) *RiskAnalysisOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RiskAnalysisOperationsListCall) Fields(s ...googleapi.Field) *RiskAnalysisOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RiskAnalysisOperationsListCall) IfNoneMatch(entityTag string) *RiskAnalysisOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RiskAnalysisOperationsListCall) Context(ctx context.Context) *RiskAnalysisOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RiskAnalysisOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RiskAnalysisOperationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.riskAnalysis.operations.list" call.
// Exactly one of *GoogleLongrunningListOperationsResponse or error will
// be non-nil. Any non-2xx status code is an error. Response headers are
// in either
// *GoogleLongrunningListOperationsResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RiskAnalysisOperationsListCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningListOperationsResponse, error) {
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
	ret := &GoogleLongrunningListOperationsResponse{
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
	//   "description": "Fetch the list of long running operations.",
	//   "flatPath": "v2beta1/riskAnalysis/operations",
	//   "httpMethod": "GET",
	//   "id": "dlp.riskAnalysis.operations.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "This parameter supports filtering by done, ie done=true or done=false.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The name of the operation's parent resource.",
	//       "location": "path",
	//       "pattern": "^riskAnalysis/operations$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The list page size. The max allowed value is 256 and default is 100.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The standard list page token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "GoogleLongrunningListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *RiskAnalysisOperationsListCall) Pages(ctx context.Context, f func(*GoogleLongrunningListOperationsResponse) error) error {
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

// method id "dlp.rootCategories.list":

type RootCategoriesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Returns the list of root categories of sensitive information.
func (r *RootCategoriesService) List() *RootCategoriesListCall {
	c := &RootCategoriesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// LanguageCode sets the optional parameter "languageCode": Optional
// language code for localized friendly category names.
// If omitted or if localized strings are not available,
// en-US strings will be returned.
func (c *RootCategoriesListCall) LanguageCode(languageCode string) *RootCategoriesListCall {
	c.urlParams_.Set("languageCode", languageCode)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RootCategoriesListCall) Fields(s ...googleapi.Field) *RootCategoriesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RootCategoriesListCall) IfNoneMatch(entityTag string) *RootCategoriesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RootCategoriesListCall) Context(ctx context.Context) *RootCategoriesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RootCategoriesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RootCategoriesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/rootCategories")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.rootCategories.list" call.
// Exactly one of *GooglePrivacyDlpV2beta1ListRootCategoriesResponse or
// error will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *GooglePrivacyDlpV2beta1ListRootCategoriesResponse.ServerResponse.Head
// er or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *RootCategoriesListCall) Do(opts ...googleapi.CallOption) (*GooglePrivacyDlpV2beta1ListRootCategoriesResponse, error) {
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
	ret := &GooglePrivacyDlpV2beta1ListRootCategoriesResponse{
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
	//   "description": "Returns the list of root categories of sensitive information.",
	//   "flatPath": "v2beta1/rootCategories",
	//   "httpMethod": "GET",
	//   "id": "dlp.rootCategories.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "languageCode": {
	//       "description": "Optional language code for localized friendly category names.\nIf omitted or if localized strings are not available,\nen-US strings will be returned.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/rootCategories",
	//   "response": {
	//     "$ref": "GooglePrivacyDlpV2beta1ListRootCategoriesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dlp.rootCategories.infoTypes.list":

type RootCategoriesInfoTypesListCall struct {
	s            *Service
	category     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Returns sensitive information types for given category.
func (r *RootCategoriesInfoTypesService) List(category string) *RootCategoriesInfoTypesListCall {
	c := &RootCategoriesInfoTypesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.category = category
	return c
}

// LanguageCode sets the optional parameter "languageCode": Optional
// BCP-47 language code for localized info type friendly
// names. If omitted, or if localized strings are not available,
// en-US strings will be returned.
func (c *RootCategoriesInfoTypesListCall) LanguageCode(languageCode string) *RootCategoriesInfoTypesListCall {
	c.urlParams_.Set("languageCode", languageCode)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RootCategoriesInfoTypesListCall) Fields(s ...googleapi.Field) *RootCategoriesInfoTypesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RootCategoriesInfoTypesListCall) IfNoneMatch(entityTag string) *RootCategoriesInfoTypesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RootCategoriesInfoTypesListCall) Context(ctx context.Context) *RootCategoriesInfoTypesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RootCategoriesInfoTypesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RootCategoriesInfoTypesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/rootCategories/{+category}/infoTypes")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"category": c.category,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dlp.rootCategories.infoTypes.list" call.
// Exactly one of *GooglePrivacyDlpV2beta1ListInfoTypesResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *GooglePrivacyDlpV2beta1ListInfoTypesResponse.ServerResponse.Header
// or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *RootCategoriesInfoTypesListCall) Do(opts ...googleapi.CallOption) (*GooglePrivacyDlpV2beta1ListInfoTypesResponse, error) {
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
	ret := &GooglePrivacyDlpV2beta1ListInfoTypesResponse{
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
	//   "description": "Returns sensitive information types for given category.",
	//   "flatPath": "v2beta1/rootCategories/{rootCategoriesId}/infoTypes",
	//   "httpMethod": "GET",
	//   "id": "dlp.rootCategories.infoTypes.list",
	//   "parameterOrder": [
	//     "category"
	//   ],
	//   "parameters": {
	//     "category": {
	//       "description": "Category name as returned by ListRootCategories.",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "languageCode": {
	//       "description": "Optional BCP-47 language code for localized info type friendly\nnames. If omitted, or if localized strings are not available,\nen-US strings will be returned.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/rootCategories/{+category}/infoTypes",
	//   "response": {
	//     "$ref": "GooglePrivacyDlpV2beta1ListInfoTypesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
