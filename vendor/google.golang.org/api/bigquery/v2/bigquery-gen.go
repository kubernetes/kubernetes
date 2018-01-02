// Package bigquery provides access to the BigQuery API.
//
// See https://cloud.google.com/bigquery/
//
// Usage example:
//
//   import "google.golang.org/api/bigquery/v2"
//   ...
//   bigqueryService, err := bigquery.New(oauthHttpClient)
package bigquery // import "google.golang.org/api/bigquery/v2"

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

const apiId = "bigquery:v2"
const apiName = "bigquery"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/bigquery/v2/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data in Google BigQuery
	BigqueryScope = "https://www.googleapis.com/auth/bigquery"

	// Insert data into Google BigQuery
	BigqueryInsertdataScope = "https://www.googleapis.com/auth/bigquery.insertdata"

	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// Manage your data and permissions in Google Cloud Storage
	DevstorageFullControlScope = "https://www.googleapis.com/auth/devstorage.full_control"

	// View your data in Google Cloud Storage
	DevstorageReadOnlyScope = "https://www.googleapis.com/auth/devstorage.read_only"

	// Manage your data in Google Cloud Storage
	DevstorageReadWriteScope = "https://www.googleapis.com/auth/devstorage.read_write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Datasets = NewDatasetsService(s)
	s.Jobs = NewJobsService(s)
	s.Projects = NewProjectsService(s)
	s.Tabledata = NewTabledataService(s)
	s.Tables = NewTablesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Datasets *DatasetsService

	Jobs *JobsService

	Projects *ProjectsService

	Tabledata *TabledataService

	Tables *TablesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewDatasetsService(s *Service) *DatasetsService {
	rs := &DatasetsService{s: s}
	return rs
}

type DatasetsService struct {
	s *Service
}

func NewJobsService(s *Service) *JobsService {
	rs := &JobsService{s: s}
	return rs
}

type JobsService struct {
	s *Service
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	return rs
}

type ProjectsService struct {
	s *Service
}

func NewTabledataService(s *Service) *TabledataService {
	rs := &TabledataService{s: s}
	return rs
}

type TabledataService struct {
	s *Service
}

func NewTablesService(s *Service) *TablesService {
	rs := &TablesService{s: s}
	return rs
}

type TablesService struct {
	s *Service
}

type BigtableColumn struct {
	// Encoding: [Optional] The encoding of the values when the type is not
	// STRING. Acceptable encoding values are: TEXT - indicates values are
	// alphanumeric text strings. BINARY - indicates values are encoded
	// using HBase Bytes.toBytes family of functions. 'encoding' can also be
	// set at the column family level. However, the setting at this level
	// takes precedence if 'encoding' is set at both levels.
	Encoding string `json:"encoding,omitempty"`

	// FieldName: [Optional] If the qualifier is not a valid BigQuery field
	// identifier i.e. does not match [a-zA-Z][a-zA-Z0-9_]*, a valid
	// identifier must be provided as the column field name and is used as
	// field name in queries.
	FieldName string `json:"fieldName,omitempty"`

	// OnlyReadLatest: [Optional] If this is set, only the latest version of
	// value in this column are exposed. 'onlyReadLatest' can also be set at
	// the column family level. However, the setting at this level takes
	// precedence if 'onlyReadLatest' is set at both levels.
	OnlyReadLatest bool `json:"onlyReadLatest,omitempty"`

	// QualifierEncoded: [Required] Qualifier of the column. Columns in the
	// parent column family that has this exact qualifier are exposed as .
	// field. If the qualifier is valid UTF-8 string, it can be specified in
	// the qualifier_string field. Otherwise, a base-64 encoded value must
	// be set to qualifier_encoded. The column field name is the same as the
	// column qualifier. However, if the qualifier is not a valid BigQuery
	// field identifier i.e. does not match [a-zA-Z][a-zA-Z0-9_]*, a valid
	// identifier must be provided as field_name.
	QualifierEncoded string `json:"qualifierEncoded,omitempty"`

	QualifierString string `json:"qualifierString,omitempty"`

	// Type: [Optional] The type to convert the value in cells of this
	// column. The values are expected to be encoded using HBase
	// Bytes.toBytes function when using the BINARY encoding value.
	// Following BigQuery types are allowed (case-sensitive) - BYTES STRING
	// INTEGER FLOAT BOOLEAN Default type is BYTES. 'type' can also be set
	// at the column family level. However, the setting at this level takes
	// precedence if 'type' is set at both levels.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Encoding") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Encoding") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BigtableColumn) MarshalJSON() ([]byte, error) {
	type noMethod BigtableColumn
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BigtableColumnFamily struct {
	// Columns: [Optional] Lists of columns that should be exposed as
	// individual fields as opposed to a list of (column name, value) pairs.
	// All columns whose qualifier matches a qualifier in this list can be
	// accessed as .. Other columns can be accessed as a list through
	// .Column field.
	Columns []*BigtableColumn `json:"columns,omitempty"`

	// Encoding: [Optional] The encoding of the values when the type is not
	// STRING. Acceptable encoding values are: TEXT - indicates values are
	// alphanumeric text strings. BINARY - indicates values are encoded
	// using HBase Bytes.toBytes family of functions. This can be overridden
	// for a specific column by listing that column in 'columns' and
	// specifying an encoding for it.
	Encoding string `json:"encoding,omitempty"`

	// FamilyId: Identifier of the column family.
	FamilyId string `json:"familyId,omitempty"`

	// OnlyReadLatest: [Optional] If this is set only the latest version of
	// value are exposed for all columns in this column family. This can be
	// overridden for a specific column by listing that column in 'columns'
	// and specifying a different setting for that column.
	OnlyReadLatest bool `json:"onlyReadLatest,omitempty"`

	// Type: [Optional] The type to convert the value in cells of this
	// column family. The values are expected to be encoded using HBase
	// Bytes.toBytes function when using the BINARY encoding value.
	// Following BigQuery types are allowed (case-sensitive) - BYTES STRING
	// INTEGER FLOAT BOOLEAN Default type is BYTES. This can be overridden
	// for a specific column by listing that column in 'columns' and
	// specifying a type for it.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Columns") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Columns") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BigtableColumnFamily) MarshalJSON() ([]byte, error) {
	type noMethod BigtableColumnFamily
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BigtableOptions struct {
	// ColumnFamilies: [Optional] List of column families to expose in the
	// table schema along with their types. This list restricts the column
	// families that can be referenced in queries and specifies their value
	// types. You can use this list to do type conversions - see the 'type'
	// field for more details. If you leave this list empty, all column
	// families are present in the table schema and their values are read as
	// BYTES. During a query only the column families referenced in that
	// query are read from Bigtable.
	ColumnFamilies []*BigtableColumnFamily `json:"columnFamilies,omitempty"`

	// IgnoreUnspecifiedColumnFamilies: [Optional] If field is true, then
	// the column families that are not specified in columnFamilies list are
	// not exposed in the table schema. Otherwise, they are read with BYTES
	// type values. The default value is false.
	IgnoreUnspecifiedColumnFamilies bool `json:"ignoreUnspecifiedColumnFamilies,omitempty"`

	// ReadRowkeyAsString: [Optional] If field is true, then the rowkey
	// column families will be read and converted to string. Otherwise they
	// are read with BYTES type values and users need to manually cast them
	// with CAST if necessary. The default value is false.
	ReadRowkeyAsString bool `json:"readRowkeyAsString,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnFamilies") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnFamilies") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BigtableOptions) MarshalJSON() ([]byte, error) {
	type noMethod BigtableOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type CsvOptions struct {
	// AllowJaggedRows: [Optional] Indicates if BigQuery should accept rows
	// that are missing trailing optional columns. If true, BigQuery treats
	// missing trailing columns as null values. If false, records with
	// missing trailing columns are treated as bad records, and if there are
	// too many bad records, an invalid error is returned in the job result.
	// The default value is false.
	AllowJaggedRows bool `json:"allowJaggedRows,omitempty"`

	// AllowQuotedNewlines: [Optional] Indicates if BigQuery should allow
	// quoted data sections that contain newline characters in a CSV file.
	// The default value is false.
	AllowQuotedNewlines bool `json:"allowQuotedNewlines,omitempty"`

	// Encoding: [Optional] The character encoding of the data. The
	// supported values are UTF-8 or ISO-8859-1. The default value is UTF-8.
	// BigQuery decodes the data after the raw, binary data has been split
	// using the values of the quote and fieldDelimiter properties.
	Encoding string `json:"encoding,omitempty"`

	// FieldDelimiter: [Optional] The separator for fields in a CSV file.
	// BigQuery converts the string to ISO-8859-1 encoding, and then uses
	// the first byte of the encoded string to split the data in its raw,
	// binary state. BigQuery also supports the escape sequence "\t" to
	// specify a tab separator. The default value is a comma (',').
	FieldDelimiter string `json:"fieldDelimiter,omitempty"`

	// Quote: [Optional] The value that is used to quote data sections in a
	// CSV file. BigQuery converts the string to ISO-8859-1 encoding, and
	// then uses the first byte of the encoded string to split the data in
	// its raw, binary state. The default value is a double-quote ('"'). If
	// your data does not contain quoted sections, set the property value to
	// an empty string. If your data contains quoted newline characters, you
	// must also set the allowQuotedNewlines property to true.
	//
	// Default: "
	Quote *string `json:"quote,omitempty"`

	// SkipLeadingRows: [Optional] The number of rows at the top of a CSV
	// file that BigQuery will skip when reading the data. The default value
	// is 0. This property is useful if you have header rows in the file
	// that should be skipped.
	SkipLeadingRows int64 `json:"skipLeadingRows,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "AllowJaggedRows") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowJaggedRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CsvOptions) MarshalJSON() ([]byte, error) {
	type noMethod CsvOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Dataset struct {
	// Access: [Optional] An array of objects that define dataset access for
	// one or more entities. You can set this property when inserting or
	// updating a dataset in order to control who is allowed to access the
	// data. If unspecified at dataset creation time, BigQuery adds default
	// dataset access for the following entities: access.specialGroup:
	// projectReaders; access.role: READER; access.specialGroup:
	// projectWriters; access.role: WRITER; access.specialGroup:
	// projectOwners; access.role: OWNER; access.userByEmail: [dataset
	// creator email]; access.role: OWNER;
	Access []*DatasetAccess `json:"access,omitempty"`

	// CreationTime: [Output-only] The time when this dataset was created,
	// in milliseconds since the epoch.
	CreationTime int64 `json:"creationTime,omitempty,string"`

	// DatasetReference: [Required] A reference that identifies the dataset.
	DatasetReference *DatasetReference `json:"datasetReference,omitempty"`

	// DefaultTableExpirationMs: [Optional] The default lifetime of all
	// tables in the dataset, in milliseconds. The minimum value is 3600000
	// milliseconds (one hour). Once this property is set, all newly-created
	// tables in the dataset will have an expirationTime property set to the
	// creation time plus the value in this property, and changing the value
	// will only affect new tables, not existing ones. When the
	// expirationTime for a given table is reached, that table will be
	// deleted automatically. If a table's expirationTime is modified or
	// removed before the table expires, or if you provide an explicit
	// expirationTime when creating a table, that value takes precedence
	// over the default expiration time indicated by this property.
	DefaultTableExpirationMs int64 `json:"defaultTableExpirationMs,omitempty,string"`

	// Description: [Optional] A user-friendly description of the dataset.
	Description string `json:"description,omitempty"`

	// Etag: [Output-only] A hash of the resource.
	Etag string `json:"etag,omitempty"`

	// FriendlyName: [Optional] A descriptive name for the dataset.
	FriendlyName string `json:"friendlyName,omitempty"`

	// Id: [Output-only] The fully-qualified unique name of the dataset in
	// the format projectId:datasetId. The dataset name without the project
	// name is given in the datasetId field. When creating a new dataset,
	// leave this field blank, and instead specify the datasetId field.
	Id string `json:"id,omitempty"`

	// Kind: [Output-only] The resource type.
	Kind string `json:"kind,omitempty"`

	// Labels: The labels associated with this dataset. You can use these to
	// organize and group your datasets. You can set this property when
	// inserting or updating a dataset. See Labeling Datasets for more
	// information.
	Labels map[string]string `json:"labels,omitempty"`

	// LastModifiedTime: [Output-only] The date when this dataset or any of
	// its tables was last modified, in milliseconds since the epoch.
	LastModifiedTime int64 `json:"lastModifiedTime,omitempty,string"`

	// Location: The geographic location where the dataset should reside.
	// Possible values include EU and US. The default value is US.
	Location string `json:"location,omitempty"`

	// SelfLink: [Output-only] A URL that can be used to access the resource
	// again. You can use this URL in Get or Update requests to the
	// resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Access") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Access") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Dataset) MarshalJSON() ([]byte, error) {
	type noMethod Dataset
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatasetAccess struct {
	// Domain: [Pick one] A domain to grant access to. Any users signed in
	// with the domain specified will be granted the specified access.
	// Example: "example.com".
	Domain string `json:"domain,omitempty"`

	// GroupByEmail: [Pick one] An email address of a Google Group to grant
	// access to.
	GroupByEmail string `json:"groupByEmail,omitempty"`

	// Role: [Required] Describes the rights granted to the user specified
	// by the other member of the access object. The following string values
	// are supported: READER, WRITER, OWNER.
	Role string `json:"role,omitempty"`

	// SpecialGroup: [Pick one] A special group to grant access to. Possible
	// values include: projectOwners: Owners of the enclosing project.
	// projectReaders: Readers of the enclosing project. projectWriters:
	// Writers of the enclosing project. allAuthenticatedUsers: All
	// authenticated BigQuery users.
	SpecialGroup string `json:"specialGroup,omitempty"`

	// UserByEmail: [Pick one] An email address of a user to grant access
	// to. For example: fred@example.com.
	UserByEmail string `json:"userByEmail,omitempty"`

	// View: [Pick one] A view from a different dataset to grant access to.
	// Queries executed against that view will have read access to tables in
	// this dataset. The role field is not required when this field is set.
	// If that view is updated by any user, access to the view needs to be
	// granted again via an update operation.
	View *TableReference `json:"view,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Domain") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Domain") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatasetAccess) MarshalJSON() ([]byte, error) {
	type noMethod DatasetAccess
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatasetList struct {
	// Datasets: An array of the dataset resources in the project. Each
	// resource contains basic information. For full information about a
	// particular dataset resource, use the Datasets: get method. This
	// property is omitted when there are no datasets in the project.
	Datasets []*DatasetListDatasets `json:"datasets,omitempty"`

	// Etag: A hash value of the results page. You can use this property to
	// determine if the page has changed since the last request.
	Etag string `json:"etag,omitempty"`

	// Kind: The list type. This property always returns the value
	// "bigquery#datasetList".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token that can be used to request the next results
	// page. This property is omitted on the final results page.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Datasets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Datasets") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatasetList) MarshalJSON() ([]byte, error) {
	type noMethod DatasetList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatasetListDatasets struct {
	// DatasetReference: The dataset reference. Use this property to access
	// specific parts of the dataset's ID, such as project ID or dataset ID.
	DatasetReference *DatasetReference `json:"datasetReference,omitempty"`

	// FriendlyName: A descriptive name for the dataset, if one exists.
	FriendlyName string `json:"friendlyName,omitempty"`

	// Id: The fully-qualified, unique, opaque ID of the dataset.
	Id string `json:"id,omitempty"`

	// Kind: The resource type. This property always returns the value
	// "bigquery#dataset".
	Kind string `json:"kind,omitempty"`

	// Labels: The labels associated with this dataset. You can use these to
	// organize and group your datasets.
	Labels map[string]string `json:"labels,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetReference") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetReference") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DatasetListDatasets) MarshalJSON() ([]byte, error) {
	type noMethod DatasetListDatasets
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type DatasetReference struct {
	// DatasetId: [Required] A unique ID for this dataset, without the
	// project name. The ID must contain only letters (a-z, A-Z), numbers
	// (0-9), or underscores (_). The maximum length is 1,024 characters.
	DatasetId string `json:"datasetId,omitempty"`

	// ProjectId: [Optional] The ID of the project containing this dataset.
	ProjectId string `json:"projectId,omitempty"`

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

func (s *DatasetReference) MarshalJSON() ([]byte, error) {
	type noMethod DatasetReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type EncryptionConfiguration struct {
	// KmsKeyName: [Optional] Describes the Cloud KMS encryption key that
	// will be used to protect destination BigQuery table. The BigQuery
	// Service Account associated with your project requires access to this
	// encryption key.
	KmsKeyName string `json:"kmsKeyName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "KmsKeyName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "KmsKeyName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EncryptionConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod EncryptionConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ErrorProto struct {
	// DebugInfo: Debugging information. This property is internal to Google
	// and should not be used.
	DebugInfo string `json:"debugInfo,omitempty"`

	// Location: Specifies where the error occurred, if present.
	Location string `json:"location,omitempty"`

	// Message: A human-readable description of the error.
	Message string `json:"message,omitempty"`

	// Reason: A short error code that summarizes the error.
	Reason string `json:"reason,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DebugInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DebugInfo") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ErrorProto) MarshalJSON() ([]byte, error) {
	type noMethod ErrorProto
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ExplainQueryStage struct {
	// ComputeMsAvg: Milliseconds the average shard spent on CPU-bound
	// tasks.
	ComputeMsAvg int64 `json:"computeMsAvg,omitempty,string"`

	// ComputeMsMax: Milliseconds the slowest shard spent on CPU-bound
	// tasks.
	ComputeMsMax int64 `json:"computeMsMax,omitempty,string"`

	// ComputeRatioAvg: Relative amount of time the average shard spent on
	// CPU-bound tasks.
	ComputeRatioAvg float64 `json:"computeRatioAvg,omitempty"`

	// ComputeRatioMax: Relative amount of time the slowest shard spent on
	// CPU-bound tasks.
	ComputeRatioMax float64 `json:"computeRatioMax,omitempty"`

	// Id: Unique ID for stage within plan.
	Id int64 `json:"id,omitempty,string"`

	// Name: Human-readable name for stage.
	Name string `json:"name,omitempty"`

	// ReadMsAvg: Milliseconds the average shard spent reading input.
	ReadMsAvg int64 `json:"readMsAvg,omitempty,string"`

	// ReadMsMax: Milliseconds the slowest shard spent reading input.
	ReadMsMax int64 `json:"readMsMax,omitempty,string"`

	// ReadRatioAvg: Relative amount of time the average shard spent reading
	// input.
	ReadRatioAvg float64 `json:"readRatioAvg,omitempty"`

	// ReadRatioMax: Relative amount of time the slowest shard spent reading
	// input.
	ReadRatioMax float64 `json:"readRatioMax,omitempty"`

	// RecordsRead: Number of records read into the stage.
	RecordsRead int64 `json:"recordsRead,omitempty,string"`

	// RecordsWritten: Number of records written by the stage.
	RecordsWritten int64 `json:"recordsWritten,omitempty,string"`

	// ShuffleOutputBytes: Total number of bytes written to shuffle.
	ShuffleOutputBytes int64 `json:"shuffleOutputBytes,omitempty,string"`

	// ShuffleOutputBytesSpilled: Total number of bytes written to shuffle
	// and spilled to disk.
	ShuffleOutputBytesSpilled int64 `json:"shuffleOutputBytesSpilled,omitempty,string"`

	// Status: Current status for the stage.
	Status string `json:"status,omitempty"`

	// Steps: List of operations within the stage in dependency order
	// (approximately chronological).
	Steps []*ExplainQueryStep `json:"steps,omitempty"`

	// WaitMsAvg: Milliseconds the average shard spent waiting to be
	// scheduled.
	WaitMsAvg int64 `json:"waitMsAvg,omitempty,string"`

	// WaitMsMax: Milliseconds the slowest shard spent waiting to be
	// scheduled.
	WaitMsMax int64 `json:"waitMsMax,omitempty,string"`

	// WaitRatioAvg: Relative amount of time the average shard spent waiting
	// to be scheduled.
	WaitRatioAvg float64 `json:"waitRatioAvg,omitempty"`

	// WaitRatioMax: Relative amount of time the slowest shard spent waiting
	// to be scheduled.
	WaitRatioMax float64 `json:"waitRatioMax,omitempty"`

	// WriteMsAvg: Milliseconds the average shard spent on writing output.
	WriteMsAvg int64 `json:"writeMsAvg,omitempty,string"`

	// WriteMsMax: Milliseconds the slowest shard spent on writing output.
	WriteMsMax int64 `json:"writeMsMax,omitempty,string"`

	// WriteRatioAvg: Relative amount of time the average shard spent on
	// writing output.
	WriteRatioAvg float64 `json:"writeRatioAvg,omitempty"`

	// WriteRatioMax: Relative amount of time the slowest shard spent on
	// writing output.
	WriteRatioMax float64 `json:"writeRatioMax,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ComputeMsAvg") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ComputeMsAvg") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExplainQueryStage) MarshalJSON() ([]byte, error) {
	type noMethod ExplainQueryStage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ExplainQueryStage) UnmarshalJSON(data []byte) error {
	type noMethod ExplainQueryStage
	var s1 struct {
		ComputeRatioAvg gensupport.JSONFloat64 `json:"computeRatioAvg"`
		ComputeRatioMax gensupport.JSONFloat64 `json:"computeRatioMax"`
		ReadRatioAvg    gensupport.JSONFloat64 `json:"readRatioAvg"`
		ReadRatioMax    gensupport.JSONFloat64 `json:"readRatioMax"`
		WaitRatioAvg    gensupport.JSONFloat64 `json:"waitRatioAvg"`
		WaitRatioMax    gensupport.JSONFloat64 `json:"waitRatioMax"`
		WriteRatioAvg   gensupport.JSONFloat64 `json:"writeRatioAvg"`
		WriteRatioMax   gensupport.JSONFloat64 `json:"writeRatioMax"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.ComputeRatioAvg = float64(s1.ComputeRatioAvg)
	s.ComputeRatioMax = float64(s1.ComputeRatioMax)
	s.ReadRatioAvg = float64(s1.ReadRatioAvg)
	s.ReadRatioMax = float64(s1.ReadRatioMax)
	s.WaitRatioAvg = float64(s1.WaitRatioAvg)
	s.WaitRatioMax = float64(s1.WaitRatioMax)
	s.WriteRatioAvg = float64(s1.WriteRatioAvg)
	s.WriteRatioMax = float64(s1.WriteRatioMax)
	return nil
}

type ExplainQueryStep struct {
	// Kind: Machine-readable operation type.
	Kind string `json:"kind,omitempty"`

	// Substeps: Human-readable stage descriptions.
	Substeps []string `json:"substeps,omitempty"`

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

func (s *ExplainQueryStep) MarshalJSON() ([]byte, error) {
	type noMethod ExplainQueryStep
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ExternalDataConfiguration struct {
	// Autodetect: Try to detect schema and format options automatically.
	// Any option specified explicitly will be honored.
	Autodetect bool `json:"autodetect,omitempty"`

	// BigtableOptions: [Optional] Additional options if sourceFormat is set
	// to BIGTABLE.
	BigtableOptions *BigtableOptions `json:"bigtableOptions,omitempty"`

	// Compression: [Optional] The compression type of the data source.
	// Possible values include GZIP and NONE. The default value is NONE.
	// This setting is ignored for Google Cloud Bigtable, Google Cloud
	// Datastore backups and Avro formats.
	Compression string `json:"compression,omitempty"`

	// CsvOptions: Additional properties to set if sourceFormat is set to
	// CSV.
	CsvOptions *CsvOptions `json:"csvOptions,omitempty"`

	// GoogleSheetsOptions: [Optional] Additional options if sourceFormat is
	// set to GOOGLE_SHEETS.
	GoogleSheetsOptions *GoogleSheetsOptions `json:"googleSheetsOptions,omitempty"`

	// IgnoreUnknownValues: [Optional] Indicates if BigQuery should allow
	// extra values that are not represented in the table schema. If true,
	// the extra values are ignored. If false, records with extra columns
	// are treated as bad records, and if there are too many bad records, an
	// invalid error is returned in the job result. The default value is
	// false. The sourceFormat property determines what BigQuery treats as
	// an extra value: CSV: Trailing columns JSON: Named values that don't
	// match any column names Google Cloud Bigtable: This setting is
	// ignored. Google Cloud Datastore backups: This setting is ignored.
	// Avro: This setting is ignored.
	IgnoreUnknownValues bool `json:"ignoreUnknownValues,omitempty"`

	// MaxBadRecords: [Optional] The maximum number of bad records that
	// BigQuery can ignore when reading data. If the number of bad records
	// exceeds this value, an invalid error is returned in the job result.
	// The default value is 0, which requires that all records are valid.
	// This setting is ignored for Google Cloud Bigtable, Google Cloud
	// Datastore backups and Avro formats.
	MaxBadRecords int64 `json:"maxBadRecords,omitempty"`

	// Schema: [Optional] The schema for the data. Schema is required for
	// CSV and JSON formats. Schema is disallowed for Google Cloud Bigtable,
	// Cloud Datastore backups, and Avro formats.
	Schema *TableSchema `json:"schema,omitempty"`

	// SourceFormat: [Required] The data format. For CSV files, specify
	// "CSV". For Google sheets, specify "GOOGLE_SHEETS". For
	// newline-delimited JSON, specify "NEWLINE_DELIMITED_JSON". For Avro
	// files, specify "AVRO". For Google Cloud Datastore backups, specify
	// "DATASTORE_BACKUP". [Beta] For Google Cloud Bigtable, specify
	// "BIGTABLE".
	SourceFormat string `json:"sourceFormat,omitempty"`

	// SourceUris: [Required] The fully-qualified URIs that point to your
	// data in Google Cloud. For Google Cloud Storage URIs: Each URI can
	// contain one '*' wildcard character and it must come after the
	// 'bucket' name. Size limits related to load jobs apply to external
	// data sources. For Google Cloud Bigtable URIs: Exactly one URI can be
	// specified and it has be a fully specified and valid HTTPS URL for a
	// Google Cloud Bigtable table. For Google Cloud Datastore backups,
	// exactly one URI can be specified. Also, the '*' wildcard character is
	// not allowed.
	SourceUris []string `json:"sourceUris,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Autodetect") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Autodetect") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExternalDataConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod ExternalDataConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GetQueryResultsResponse struct {
	// CacheHit: Whether the query result was fetched from the query cache.
	CacheHit bool `json:"cacheHit,omitempty"`

	// Errors: [Output-only] The first errors or warnings encountered during
	// the running of the job. The final message includes the number of
	// errors that caused the process to stop. Errors here do not
	// necessarily mean that the job has completed or was unsuccessful.
	Errors []*ErrorProto `json:"errors,omitempty"`

	// Etag: A hash of this response.
	Etag string `json:"etag,omitempty"`

	// JobComplete: Whether the query has completed or not. If rows or
	// totalRows are present, this will always be true. If this is false,
	// totalRows will not be available.
	JobComplete bool `json:"jobComplete,omitempty"`

	// JobReference: Reference to the BigQuery Job that was created to run
	// the query. This field will be present even if the original request
	// timed out, in which case GetQueryResults can be used to read the
	// results once the query has completed. Since this API only returns the
	// first page of results, subsequent pages can be fetched via the same
	// mechanism (GetQueryResults).
	JobReference *JobReference `json:"jobReference,omitempty"`

	// Kind: The resource type of the response.
	Kind string `json:"kind,omitempty"`

	// NumDmlAffectedRows: [Output-only] The number of rows affected by a
	// DML statement. Present only for DML statements INSERT, UPDATE or
	// DELETE.
	NumDmlAffectedRows int64 `json:"numDmlAffectedRows,omitempty,string"`

	// PageToken: A token used for paging results.
	PageToken string `json:"pageToken,omitempty"`

	// Rows: An object with as many results as can be contained within the
	// maximum permitted reply size. To get any additional rows, you can
	// call GetQueryResults and specify the jobReference returned above.
	// Present only when the query completes successfully.
	Rows []*TableRow `json:"rows,omitempty"`

	// Schema: The schema of the results. Present only when the query
	// completes successfully.
	Schema *TableSchema `json:"schema,omitempty"`

	// TotalBytesProcessed: The total number of bytes processed for this
	// query.
	TotalBytesProcessed int64 `json:"totalBytesProcessed,omitempty,string"`

	// TotalRows: The total number of rows in the complete query result set,
	// which can be more than the number of rows in this single page of
	// results. Present only when the query completes successfully.
	TotalRows uint64 `json:"totalRows,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CacheHit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CacheHit") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GetQueryResultsResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetQueryResultsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GetServiceAccountResponse struct {
	// Email: The service account email address.
	Email string `json:"email,omitempty"`

	// Kind: The resource type of the response.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Email") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Email") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GetServiceAccountResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetServiceAccountResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GoogleSheetsOptions struct {
	// SkipLeadingRows: [Optional] The number of rows at the top of a sheet
	// that BigQuery will skip when reading the data. The default value is
	// 0. This property is useful if you have header rows that should be
	// skipped. When autodetect is on, behavior is the following: *
	// skipLeadingRows unspecified - Autodetect tries to detect headers in
	// the first row. If they are not detected, the row is read as data.
	// Otherwise data is read starting from the second row. *
	// skipLeadingRows is 0 - Instructs autodetect that there are no headers
	// and data should be read starting from the first row. *
	// skipLeadingRows = N > 0 - Autodetect skips N-1 rows and tries to
	// detect headers in row N. If headers are not detected, row N is just
	// skipped. Otherwise row N is used to extract column names for the
	// detected schema.
	SkipLeadingRows int64 `json:"skipLeadingRows,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "SkipLeadingRows") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SkipLeadingRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleSheetsOptions) MarshalJSON() ([]byte, error) {
	type noMethod GoogleSheetsOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Job struct {
	// Configuration: [Required] Describes the job configuration.
	Configuration *JobConfiguration `json:"configuration,omitempty"`

	// Etag: [Output-only] A hash of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: [Output-only] Opaque ID field of the job
	Id string `json:"id,omitempty"`

	// JobReference: [Optional] Reference describing the unique-per-user
	// name of the job.
	JobReference *JobReference `json:"jobReference,omitempty"`

	// Kind: [Output-only] The type of the resource.
	Kind string `json:"kind,omitempty"`

	// SelfLink: [Output-only] A URL that can be used to access this
	// resource again.
	SelfLink string `json:"selfLink,omitempty"`

	// Statistics: [Output-only] Information about the job, including
	// starting time and ending time of the job.
	Statistics *JobStatistics `json:"statistics,omitempty"`

	// Status: [Output-only] The status of this job. Examine this value when
	// polling an asynchronous job to see if the job is complete.
	Status *JobStatus `json:"status,omitempty"`

	// UserEmail: [Output-only] Email address of the user who ran the job.
	UserEmail string `json:"user_email,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Configuration") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Configuration") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Job) MarshalJSON() ([]byte, error) {
	type noMethod Job
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobCancelResponse struct {
	// Job: The final state of the job.
	Job *Job `json:"job,omitempty"`

	// Kind: The resource type of the response.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Job") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Job") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobCancelResponse) MarshalJSON() ([]byte, error) {
	type noMethod JobCancelResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobConfiguration struct {
	// Copy: [Pick one] Copies a table.
	Copy *JobConfigurationTableCopy `json:"copy,omitempty"`

	// DryRun: [Optional] If set, don't actually run this job. A valid query
	// will return a mostly empty response with some processing statistics,
	// while an invalid query will return the same error it would if it
	// wasn't a dry run. Behavior of non-query jobs is undefined.
	DryRun bool `json:"dryRun,omitempty"`

	// Extract: [Pick one] Configures an extract job.
	Extract *JobConfigurationExtract `json:"extract,omitempty"`

	// Labels: [Experimental] The labels associated with this job. You can
	// use these to organize and group your jobs. Label keys and values can
	// be no longer than 63 characters, can only contain lowercase letters,
	// numeric characters, underscores and dashes. International characters
	// are allowed. Label values are optional. Label keys must start with a
	// letter and each label in the list must have a different key.
	Labels map[string]string `json:"labels,omitempty"`

	// Load: [Pick one] Configures a load job.
	Load *JobConfigurationLoad `json:"load,omitempty"`

	// Query: [Pick one] Configures a query job.
	Query *JobConfigurationQuery `json:"query,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Copy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Copy") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod JobConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobConfigurationExtract struct {
	// Compression: [Optional] The compression type to use for exported
	// files. Possible values include GZIP and NONE. The default value is
	// NONE.
	Compression string `json:"compression,omitempty"`

	// DestinationFormat: [Optional] The exported file format. Possible
	// values include CSV, NEWLINE_DELIMITED_JSON and AVRO. The default
	// value is CSV. Tables with nested or repeated fields cannot be
	// exported as CSV.
	DestinationFormat string `json:"destinationFormat,omitempty"`

	// DestinationUri: [Pick one] DEPRECATED: Use destinationUris instead,
	// passing only one URI as necessary. The fully-qualified Google Cloud
	// Storage URI where the extracted table should be written.
	DestinationUri string `json:"destinationUri,omitempty"`

	// DestinationUris: [Pick one] A list of fully-qualified Google Cloud
	// Storage URIs where the extracted table should be written.
	DestinationUris []string `json:"destinationUris,omitempty"`

	// FieldDelimiter: [Optional] Delimiter to use between fields in the
	// exported data. Default is ','
	FieldDelimiter string `json:"fieldDelimiter,omitempty"`

	// PrintHeader: [Optional] Whether to print out a header row in the
	// results. Default is true.
	//
	// Default: true
	PrintHeader *bool `json:"printHeader,omitempty"`

	// SourceTable: [Required] A reference to the table being exported.
	SourceTable *TableReference `json:"sourceTable,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Compression") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Compression") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobConfigurationExtract) MarshalJSON() ([]byte, error) {
	type noMethod JobConfigurationExtract
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobConfigurationLoad struct {
	// AllowJaggedRows: [Optional] Accept rows that are missing trailing
	// optional columns. The missing values are treated as nulls. If false,
	// records with missing trailing columns are treated as bad records, and
	// if there are too many bad records, an invalid error is returned in
	// the job result. The default value is false. Only applicable to CSV,
	// ignored for other formats.
	AllowJaggedRows bool `json:"allowJaggedRows,omitempty"`

	// AllowQuotedNewlines: Indicates if BigQuery should allow quoted data
	// sections that contain newline characters in a CSV file. The default
	// value is false.
	AllowQuotedNewlines bool `json:"allowQuotedNewlines,omitempty"`

	// Autodetect: Indicates if we should automatically infer the options
	// and schema for CSV and JSON sources.
	Autodetect bool `json:"autodetect,omitempty"`

	// CreateDisposition: [Optional] Specifies whether the job is allowed to
	// create new tables. The following values are supported:
	// CREATE_IF_NEEDED: If the table does not exist, BigQuery creates the
	// table. CREATE_NEVER: The table must already exist. If it does not, a
	// 'notFound' error is returned in the job result. The default value is
	// CREATE_IF_NEEDED. Creation, truncation and append actions occur as
	// one atomic update upon job completion.
	CreateDisposition string `json:"createDisposition,omitempty"`

	// DestinationEncryptionConfiguration: [Experimental] Custom encryption
	// configuration (e.g., Cloud KMS keys).
	DestinationEncryptionConfiguration *EncryptionConfiguration `json:"destinationEncryptionConfiguration,omitempty"`

	// DestinationTable: [Required] The destination table to load the data
	// into.
	DestinationTable *TableReference `json:"destinationTable,omitempty"`

	// Encoding: [Optional] The character encoding of the data. The
	// supported values are UTF-8 or ISO-8859-1. The default value is UTF-8.
	// BigQuery decodes the data after the raw, binary data has been split
	// using the values of the quote and fieldDelimiter properties.
	Encoding string `json:"encoding,omitempty"`

	// FieldDelimiter: [Optional] The separator for fields in a CSV file.
	// The separator can be any ISO-8859-1 single-byte character. To use a
	// character in the range 128-255, you must encode the character as
	// UTF8. BigQuery converts the string to ISO-8859-1 encoding, and then
	// uses the first byte of the encoded string to split the data in its
	// raw, binary state. BigQuery also supports the escape sequence "\t" to
	// specify a tab separator. The default value is a comma (',').
	FieldDelimiter string `json:"fieldDelimiter,omitempty"`

	// IgnoreUnknownValues: [Optional] Indicates if BigQuery should allow
	// extra values that are not represented in the table schema. If true,
	// the extra values are ignored. If false, records with extra columns
	// are treated as bad records, and if there are too many bad records, an
	// invalid error is returned in the job result. The default value is
	// false. The sourceFormat property determines what BigQuery treats as
	// an extra value: CSV: Trailing columns JSON: Named values that don't
	// match any column names
	IgnoreUnknownValues bool `json:"ignoreUnknownValues,omitempty"`

	// MaxBadRecords: [Optional] The maximum number of bad records that
	// BigQuery can ignore when running the job. If the number of bad
	// records exceeds this value, an invalid error is returned in the job
	// result. The default value is 0, which requires that all records are
	// valid.
	MaxBadRecords int64 `json:"maxBadRecords,omitempty"`

	// NullMarker: [Optional] Specifies a string that represents a null
	// value in a CSV file. For example, if you specify "\N", BigQuery
	// interprets "\N" as a null value when loading a CSV file. The default
	// value is the empty string. If you set this property to a custom
	// value, BigQuery throws an error if an empty string is present for all
	// data types except for STRING and BYTE. For STRING and BYTE columns,
	// BigQuery interprets the empty string as an empty value.
	NullMarker string `json:"nullMarker,omitempty"`

	// ProjectionFields: If sourceFormat is set to "DATASTORE_BACKUP",
	// indicates which entity properties to load into BigQuery from a Cloud
	// Datastore backup. Property names are case sensitive and must be
	// top-level properties. If no properties are specified, BigQuery loads
	// all properties. If any named property isn't found in the Cloud
	// Datastore backup, an invalid error is returned in the job result.
	ProjectionFields []string `json:"projectionFields,omitempty"`

	// Quote: [Optional] The value that is used to quote data sections in a
	// CSV file. BigQuery converts the string to ISO-8859-1 encoding, and
	// then uses the first byte of the encoded string to split the data in
	// its raw, binary state. The default value is a double-quote ('"'). If
	// your data does not contain quoted sections, set the property value to
	// an empty string. If your data contains quoted newline characters, you
	// must also set the allowQuotedNewlines property to true.
	//
	// Default: "
	Quote *string `json:"quote,omitempty"`

	// Schema: [Optional] The schema for the destination table. The schema
	// can be omitted if the destination table already exists, or if you're
	// loading data from Google Cloud Datastore.
	Schema *TableSchema `json:"schema,omitempty"`

	// SchemaInline: [Deprecated] The inline schema. For CSV schemas,
	// specify as "Field1:Type1[,Field2:Type2]*". For example, "foo:STRING,
	// bar:INTEGER, baz:FLOAT".
	SchemaInline string `json:"schemaInline,omitempty"`

	// SchemaInlineFormat: [Deprecated] The format of the schemaInline
	// property.
	SchemaInlineFormat string `json:"schemaInlineFormat,omitempty"`

	// SchemaUpdateOptions: [Experimental] Allows the schema of the
	// desitination table to be updated as a side effect of the load job if
	// a schema is autodetected or supplied in the job configuration. Schema
	// update options are supported in two cases: when writeDisposition is
	// WRITE_APPEND; when writeDisposition is WRITE_TRUNCATE and the
	// destination table is a partition of a table, specified by partition
	// decorators. For normal tables, WRITE_TRUNCATE will always overwrite
	// the schema. One or more of the following values are specified:
	// ALLOW_FIELD_ADDITION: allow adding a nullable field to the schema.
	// ALLOW_FIELD_RELAXATION: allow relaxing a required field in the
	// original schema to nullable.
	SchemaUpdateOptions []string `json:"schemaUpdateOptions,omitempty"`

	// SkipLeadingRows: [Optional] The number of rows at the top of a CSV
	// file that BigQuery will skip when loading the data. The default value
	// is 0. This property is useful if you have header rows in the file
	// that should be skipped.
	SkipLeadingRows int64 `json:"skipLeadingRows,omitempty"`

	// SourceFormat: [Optional] The format of the data files. For CSV files,
	// specify "CSV". For datastore backups, specify "DATASTORE_BACKUP". For
	// newline-delimited JSON, specify "NEWLINE_DELIMITED_JSON". For Avro,
	// specify "AVRO". The default value is CSV.
	SourceFormat string `json:"sourceFormat,omitempty"`

	// SourceUris: [Required] The fully-qualified URIs that point to your
	// data in Google Cloud. For Google Cloud Storage URIs: Each URI can
	// contain one '*' wildcard character and it must come after the
	// 'bucket' name. Size limits related to load jobs apply to external
	// data sources. For Google Cloud Bigtable URIs: Exactly one URI can be
	// specified and it has be a fully specified and valid HTTPS URL for a
	// Google Cloud Bigtable table. For Google Cloud Datastore backups:
	// Exactly one URI can be specified. Also, the '*' wildcard character is
	// not allowed.
	SourceUris []string `json:"sourceUris,omitempty"`

	// TimePartitioning: [Experimental] If specified, configures time-based
	// partitioning for the destination table.
	TimePartitioning *TimePartitioning `json:"timePartitioning,omitempty"`

	// WriteDisposition: [Optional] Specifies the action that occurs if the
	// destination table already exists. The following values are supported:
	// WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the
	// table data. WRITE_APPEND: If the table already exists, BigQuery
	// appends the data to the table. WRITE_EMPTY: If the table already
	// exists and contains data, a 'duplicate' error is returned in the job
	// result. The default value is WRITE_APPEND. Each action is atomic and
	// only occurs if BigQuery is able to complete the job successfully.
	// Creation, truncation and append actions occur as one atomic update
	// upon job completion.
	WriteDisposition string `json:"writeDisposition,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AllowJaggedRows") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowJaggedRows") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *JobConfigurationLoad) MarshalJSON() ([]byte, error) {
	type noMethod JobConfigurationLoad
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobConfigurationQuery struct {
	// AllowLargeResults: [Optional] If true and query uses legacy SQL
	// dialect, allows the query to produce arbitrarily large result tables
	// at a slight cost in performance. Requires destinationTable to be set.
	// For standard SQL queries, this flag is ignored and large results are
	// always allowed. However, you must still set destinationTable when
	// result size exceeds the allowed maximum response size.
	AllowLargeResults bool `json:"allowLargeResults,omitempty"`

	// CreateDisposition: [Optional] Specifies whether the job is allowed to
	// create new tables. The following values are supported:
	// CREATE_IF_NEEDED: If the table does not exist, BigQuery creates the
	// table. CREATE_NEVER: The table must already exist. If it does not, a
	// 'notFound' error is returned in the job result. The default value is
	// CREATE_IF_NEEDED. Creation, truncation and append actions occur as
	// one atomic update upon job completion.
	CreateDisposition string `json:"createDisposition,omitempty"`

	// DefaultDataset: [Optional] Specifies the default dataset to use for
	// unqualified table names in the query.
	DefaultDataset *DatasetReference `json:"defaultDataset,omitempty"`

	// DestinationEncryptionConfiguration: [Experimental] Custom encryption
	// configuration (e.g., Cloud KMS keys).
	DestinationEncryptionConfiguration *EncryptionConfiguration `json:"destinationEncryptionConfiguration,omitempty"`

	// DestinationTable: [Optional] Describes the table where the query
	// results should be stored. If not present, a new table will be created
	// to store the results. This property must be set for large results
	// that exceed the maximum response size.
	DestinationTable *TableReference `json:"destinationTable,omitempty"`

	// FlattenResults: [Optional] If true and query uses legacy SQL dialect,
	// flattens all nested and repeated fields in the query results.
	// allowLargeResults must be true if this is set to false. For standard
	// SQL queries, this flag is ignored and results are never flattened.
	//
	// Default: true
	FlattenResults *bool `json:"flattenResults,omitempty"`

	// MaximumBillingTier: [Optional] Limits the billing tier for this job.
	// Queries that have resource usage beyond this tier will fail (without
	// incurring a charge). If unspecified, this will be set to your project
	// default.
	//
	// Default: 1
	MaximumBillingTier *int64 `json:"maximumBillingTier,omitempty"`

	// MaximumBytesBilled: [Optional] Limits the bytes billed for this job.
	// Queries that will have bytes billed beyond this limit will fail
	// (without incurring a charge). If unspecified, this will be set to
	// your project default.
	MaximumBytesBilled int64 `json:"maximumBytesBilled,omitempty,string"`

	// ParameterMode: Standard SQL only. Set to POSITIONAL to use positional
	// (?) query parameters or to NAMED to use named (@myparam) query
	// parameters in this query.
	ParameterMode string `json:"parameterMode,omitempty"`

	// PreserveNulls: [Deprecated] This property is deprecated.
	PreserveNulls bool `json:"preserveNulls,omitempty"`

	// Priority: [Optional] Specifies a priority for the query. Possible
	// values include INTERACTIVE and BATCH. The default value is
	// INTERACTIVE.
	Priority string `json:"priority,omitempty"`

	// Query: [Required] SQL query text to execute. The useLegacySql field
	// can be used to indicate whether the query uses legacy SQL or standard
	// SQL.
	Query string `json:"query,omitempty"`

	// QueryParameters: Query parameters for standard SQL queries.
	QueryParameters []*QueryParameter `json:"queryParameters,omitempty"`

	// SchemaUpdateOptions: [Experimental] Allows the schema of the
	// destination table to be updated as a side effect of the query job.
	// Schema update options are supported in two cases: when
	// writeDisposition is WRITE_APPEND; when writeDisposition is
	// WRITE_TRUNCATE and the destination table is a partition of a table,
	// specified by partition decorators. For normal tables, WRITE_TRUNCATE
	// will always overwrite the schema. One or more of the following values
	// are specified: ALLOW_FIELD_ADDITION: allow adding a nullable field to
	// the schema. ALLOW_FIELD_RELAXATION: allow relaxing a required field
	// in the original schema to nullable.
	SchemaUpdateOptions []string `json:"schemaUpdateOptions,omitempty"`

	// TableDefinitions: [Optional] If querying an external data source
	// outside of BigQuery, describes the data format, location and other
	// properties of the data source. By defining these properties, the data
	// source can then be queried as if it were a standard BigQuery table.
	TableDefinitions map[string]ExternalDataConfiguration `json:"tableDefinitions,omitempty"`

	// TimePartitioning: [Experimental] If specified, configures time-based
	// partitioning for the destination table.
	TimePartitioning *TimePartitioning `json:"timePartitioning,omitempty"`

	// UseLegacySql: Specifies whether to use BigQuery's legacy SQL dialect
	// for this query. The default value is true. If set to false, the query
	// will use BigQuery's standard SQL:
	// https://cloud.google.com/bigquery/sql-reference/ When useLegacySql is
	// set to false, the value of flattenResults is ignored; query will be
	// run as if flattenResults is false.
	UseLegacySql bool `json:"useLegacySql,omitempty"`

	// UseQueryCache: [Optional] Whether to look for the result in the query
	// cache. The query cache is a best-effort cache that will be flushed
	// whenever tables in the query are modified. Moreover, the query cache
	// is only available when a query does not have a destination table
	// specified. The default value is true.
	//
	// Default: true
	UseQueryCache *bool `json:"useQueryCache,omitempty"`

	// UserDefinedFunctionResources: Describes user-defined function
	// resources used in the query.
	UserDefinedFunctionResources []*UserDefinedFunctionResource `json:"userDefinedFunctionResources,omitempty"`

	// WriteDisposition: [Optional] Specifies the action that occurs if the
	// destination table already exists. The following values are supported:
	// WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the
	// table data and uses the schema from the query result. WRITE_APPEND:
	// If the table already exists, BigQuery appends the data to the table.
	// WRITE_EMPTY: If the table already exists and contains data, a
	// 'duplicate' error is returned in the job result. The default value is
	// WRITE_EMPTY. Each action is atomic and only occurs if BigQuery is
	// able to complete the job successfully. Creation, truncation and
	// append actions occur as one atomic update upon job completion.
	WriteDisposition string `json:"writeDisposition,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AllowLargeResults")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowLargeResults") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *JobConfigurationQuery) MarshalJSON() ([]byte, error) {
	type noMethod JobConfigurationQuery
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobConfigurationTableCopy struct {
	// CreateDisposition: [Optional] Specifies whether the job is allowed to
	// create new tables. The following values are supported:
	// CREATE_IF_NEEDED: If the table does not exist, BigQuery creates the
	// table. CREATE_NEVER: The table must already exist. If it does not, a
	// 'notFound' error is returned in the job result. The default value is
	// CREATE_IF_NEEDED. Creation, truncation and append actions occur as
	// one atomic update upon job completion.
	CreateDisposition string `json:"createDisposition,omitempty"`

	// DestinationEncryptionConfiguration: [Experimental] Custom encryption
	// configuration (e.g., Cloud KMS keys).
	DestinationEncryptionConfiguration *EncryptionConfiguration `json:"destinationEncryptionConfiguration,omitempty"`

	// DestinationTable: [Required] The destination table
	DestinationTable *TableReference `json:"destinationTable,omitempty"`

	// SourceTable: [Pick one] Source table to copy.
	SourceTable *TableReference `json:"sourceTable,omitempty"`

	// SourceTables: [Pick one] Source tables to copy.
	SourceTables []*TableReference `json:"sourceTables,omitempty"`

	// WriteDisposition: [Optional] Specifies the action that occurs if the
	// destination table already exists. The following values are supported:
	// WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the
	// table data. WRITE_APPEND: If the table already exists, BigQuery
	// appends the data to the table. WRITE_EMPTY: If the table already
	// exists and contains data, a 'duplicate' error is returned in the job
	// result. The default value is WRITE_EMPTY. Each action is atomic and
	// only occurs if BigQuery is able to complete the job successfully.
	// Creation, truncation and append actions occur as one atomic update
	// upon job completion.
	WriteDisposition string `json:"writeDisposition,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreateDisposition")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateDisposition") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *JobConfigurationTableCopy) MarshalJSON() ([]byte, error) {
	type noMethod JobConfigurationTableCopy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobList struct {
	// Etag: A hash of this page of results.
	Etag string `json:"etag,omitempty"`

	// Jobs: List of jobs that were requested.
	Jobs []*JobListJobs `json:"jobs,omitempty"`

	// Kind: The resource type of the response.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to request the next page of results.
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

func (s *JobList) MarshalJSON() ([]byte, error) {
	type noMethod JobList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobListJobs struct {
	// Configuration: [Full-projection-only] Specifies the job
	// configuration.
	Configuration *JobConfiguration `json:"configuration,omitempty"`

	// ErrorResult: A result object that will be present only if the job has
	// failed.
	ErrorResult *ErrorProto `json:"errorResult,omitempty"`

	// Id: Unique opaque ID of the job.
	Id string `json:"id,omitempty"`

	// JobReference: Job reference uniquely identifying the job.
	JobReference *JobReference `json:"jobReference,omitempty"`

	// Kind: The resource type.
	Kind string `json:"kind,omitempty"`

	// State: Running state of the job. When the state is DONE, errorResult
	// can be checked to determine whether the job succeeded or failed.
	State string `json:"state,omitempty"`

	// Statistics: [Output-only] Information about the job, including
	// starting time and ending time of the job.
	Statistics *JobStatistics `json:"statistics,omitempty"`

	// Status: [Full-projection-only] Describes the state of the job.
	Status *JobStatus `json:"status,omitempty"`

	// UserEmail: [Full-projection-only] Email address of the user who ran
	// the job.
	UserEmail string `json:"user_email,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Configuration") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Configuration") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobListJobs) MarshalJSON() ([]byte, error) {
	type noMethod JobListJobs
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobReference struct {
	// JobId: [Required] The ID of the job. The ID must contain only letters
	// (a-z, A-Z), numbers (0-9), underscores (_), or dashes (-). The
	// maximum length is 1,024 characters.
	JobId string `json:"jobId,omitempty"`

	// ProjectId: [Required] The ID of the project containing this job.
	ProjectId string `json:"projectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "JobId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "JobId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobReference) MarshalJSON() ([]byte, error) {
	type noMethod JobReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobStatistics struct {
	// CreationTime: [Output-only] Creation time of this job, in
	// milliseconds since the epoch. This field will be present on all jobs.
	CreationTime int64 `json:"creationTime,omitempty,string"`

	// EndTime: [Output-only] End time of this job, in milliseconds since
	// the epoch. This field will be present whenever a job is in the DONE
	// state.
	EndTime int64 `json:"endTime,omitempty,string"`

	// Extract: [Output-only] Statistics for an extract job.
	Extract *JobStatistics4 `json:"extract,omitempty"`

	// Load: [Output-only] Statistics for a load job.
	Load *JobStatistics3 `json:"load,omitempty"`

	// Query: [Output-only] Statistics for a query job.
	Query *JobStatistics2 `json:"query,omitempty"`

	// StartTime: [Output-only] Start time of this job, in milliseconds
	// since the epoch. This field will be present when the job transitions
	// from the PENDING state to either RUNNING or DONE.
	StartTime int64 `json:"startTime,omitempty,string"`

	// TotalBytesProcessed: [Output-only] [Deprecated] Use the bytes
	// processed in the query statistics instead.
	TotalBytesProcessed int64 `json:"totalBytesProcessed,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "CreationTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobStatistics) MarshalJSON() ([]byte, error) {
	type noMethod JobStatistics
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobStatistics2 struct {
	// BillingTier: [Output-only] Billing tier for the job.
	BillingTier int64 `json:"billingTier,omitempty"`

	// CacheHit: [Output-only] Whether the query result was fetched from the
	// query cache.
	CacheHit bool `json:"cacheHit,omitempty"`

	// NumDmlAffectedRows: [Output-only] The number of rows affected by a
	// DML statement. Present only for DML statements INSERT, UPDATE or
	// DELETE.
	NumDmlAffectedRows int64 `json:"numDmlAffectedRows,omitempty,string"`

	// QueryPlan: [Output-only] Describes execution plan for the query.
	QueryPlan []*ExplainQueryStage `json:"queryPlan,omitempty"`

	// ReferencedTables: [Output-only, Experimental] Referenced tables for
	// the job. Queries that reference more than 50 tables will not have a
	// complete list.
	ReferencedTables []*TableReference `json:"referencedTables,omitempty"`

	// Schema: [Output-only, Experimental] The schema of the results.
	// Present only for successful dry run of non-legacy SQL queries.
	Schema *TableSchema `json:"schema,omitempty"`

	// StatementType: [Output-only, Experimental] The type of query
	// statement, if valid.
	StatementType string `json:"statementType,omitempty"`

	// TotalBytesBilled: [Output-only] Total bytes billed for the job.
	TotalBytesBilled int64 `json:"totalBytesBilled,omitempty,string"`

	// TotalBytesProcessed: [Output-only] Total bytes processed for the job.
	TotalBytesProcessed int64 `json:"totalBytesProcessed,omitempty,string"`

	// UndeclaredQueryParameters: [Output-only, Experimental] Standard SQL
	// only: list of undeclared query parameters detected during a dry run
	// validation.
	UndeclaredQueryParameters []*QueryParameter `json:"undeclaredQueryParameters,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BillingTier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BillingTier") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobStatistics2) MarshalJSON() ([]byte, error) {
	type noMethod JobStatistics2
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobStatistics3 struct {
	// BadRecords: [Output-only] The number of bad records encountered. Note
	// that if the job has failed because of more bad records encountered
	// than the maximum allowed in the load job configuration, then this
	// number can be less than the total number of bad records present in
	// the input data.
	BadRecords int64 `json:"badRecords,omitempty,string"`

	// InputFileBytes: [Output-only] Number of bytes of source data in a
	// load job.
	InputFileBytes int64 `json:"inputFileBytes,omitempty,string"`

	// InputFiles: [Output-only] Number of source files in a load job.
	InputFiles int64 `json:"inputFiles,omitempty,string"`

	// OutputBytes: [Output-only] Size of the loaded data in bytes. Note
	// that while a load job is in the running state, this value may change.
	OutputBytes int64 `json:"outputBytes,omitempty,string"`

	// OutputRows: [Output-only] Number of rows imported in a load job. Note
	// that while an import job is in the running state, this value may
	// change.
	OutputRows int64 `json:"outputRows,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "BadRecords") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BadRecords") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobStatistics3) MarshalJSON() ([]byte, error) {
	type noMethod JobStatistics3
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobStatistics4 struct {
	// DestinationUriFileCounts: [Output-only] Number of files per
	// destination URI or URI pattern specified in the extract
	// configuration. These values will be in the same order as the URIs
	// specified in the 'destinationUris' field.
	DestinationUriFileCounts googleapi.Int64s `json:"destinationUriFileCounts,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DestinationUriFileCounts") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DestinationUriFileCounts")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *JobStatistics4) MarshalJSON() ([]byte, error) {
	type noMethod JobStatistics4
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JobStatus struct {
	// ErrorResult: [Output-only] Final error result of the job. If present,
	// indicates that the job has completed and was unsuccessful.
	ErrorResult *ErrorProto `json:"errorResult,omitempty"`

	// Errors: [Output-only] The first errors encountered during the running
	// of the job. The final message includes the number of errors that
	// caused the process to stop. Errors here do not necessarily mean that
	// the job has completed or was unsuccessful.
	Errors []*ErrorProto `json:"errors,omitempty"`

	// State: [Output-only] Running state of the job.
	State string `json:"state,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ErrorResult") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ErrorResult") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobStatus) MarshalJSON() ([]byte, error) {
	type noMethod JobStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JsonValue interface{}

type ProjectList struct {
	// Etag: A hash of the page of results
	Etag string `json:"etag,omitempty"`

	// Kind: The type of list.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to request the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Projects: Projects to which you have at least READ access.
	Projects []*ProjectListProjects `json:"projects,omitempty"`

	// TotalItems: The total number of projects in the list.
	TotalItems int64 `json:"totalItems,omitempty"`

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

func (s *ProjectList) MarshalJSON() ([]byte, error) {
	type noMethod ProjectList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProjectListProjects struct {
	// FriendlyName: A descriptive name for this project.
	FriendlyName string `json:"friendlyName,omitempty"`

	// Id: An opaque ID of this project.
	Id string `json:"id,omitempty"`

	// Kind: The resource type.
	Kind string `json:"kind,omitempty"`

	// NumericId: The numeric ID of this project.
	NumericId uint64 `json:"numericId,omitempty,string"`

	// ProjectReference: A unique reference to this project.
	ProjectReference *ProjectReference `json:"projectReference,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FriendlyName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FriendlyName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProjectListProjects) MarshalJSON() ([]byte, error) {
	type noMethod ProjectListProjects
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProjectReference struct {
	// ProjectId: [Required] ID of the project. Can be either the numeric ID
	// or the assigned ID of the project.
	ProjectId string `json:"projectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProjectId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProjectId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProjectReference) MarshalJSON() ([]byte, error) {
	type noMethod ProjectReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type QueryParameter struct {
	// Name: [Optional] If unset, this is a positional parameter. Otherwise,
	// should be unique within a query.
	Name string `json:"name,omitempty"`

	// ParameterType: [Required] The type of this parameter.
	ParameterType *QueryParameterType `json:"parameterType,omitempty"`

	// ParameterValue: [Required] The value of this parameter.
	ParameterValue *QueryParameterValue `json:"parameterValue,omitempty"`

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

func (s *QueryParameter) MarshalJSON() ([]byte, error) {
	type noMethod QueryParameter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type QueryParameterType struct {
	// ArrayType: [Optional] The type of the array's elements, if this is an
	// array.
	ArrayType *QueryParameterType `json:"arrayType,omitempty"`

	// StructTypes: [Optional] The types of the fields of this struct, in
	// order, if this is a struct.
	StructTypes []*QueryParameterTypeStructTypes `json:"structTypes,omitempty"`

	// Type: [Required] The top level type of this field.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArrayType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ArrayType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryParameterType) MarshalJSON() ([]byte, error) {
	type noMethod QueryParameterType
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type QueryParameterTypeStructTypes struct {
	// Description: [Optional] Human-oriented description of the field.
	Description string `json:"description,omitempty"`

	// Name: [Optional] The name of this field.
	Name string `json:"name,omitempty"`

	// Type: [Required] The type of this field.
	Type *QueryParameterType `json:"type,omitempty"`

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

func (s *QueryParameterTypeStructTypes) MarshalJSON() ([]byte, error) {
	type noMethod QueryParameterTypeStructTypes
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type QueryParameterValue struct {
	// ArrayValues: [Optional] The array values, if this is an array type.
	ArrayValues []*QueryParameterValue `json:"arrayValues,omitempty"`

	// StructValues: [Optional] The struct field values, in order of the
	// struct type's declaration.
	StructValues map[string]QueryParameterValue `json:"structValues,omitempty"`

	// Value: [Optional] The value of this value, if a simple scalar type.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArrayValues") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ArrayValues") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryParameterValue) MarshalJSON() ([]byte, error) {
	type noMethod QueryParameterValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type QueryRequest struct {
	// DefaultDataset: [Optional] Specifies the default datasetId and
	// projectId to assume for any unqualified table names in the query. If
	// not set, all table names in the query string must be qualified in the
	// format 'datasetId.tableId'.
	DefaultDataset *DatasetReference `json:"defaultDataset,omitempty"`

	// DryRun: [Optional] If set to true, BigQuery doesn't run the job.
	// Instead, if the query is valid, BigQuery returns statistics about the
	// job such as how many bytes would be processed. If the query is
	// invalid, an error returns. The default value is false.
	DryRun bool `json:"dryRun,omitempty"`

	// Kind: The resource type of the request.
	Kind string `json:"kind,omitempty"`

	// MaxResults: [Optional] The maximum number of rows of data to return
	// per page of results. Setting this flag to a small value such as 1000
	// and then paging through results might improve reliability when the
	// query result set is large. In addition to this limit, responses are
	// also limited to 10 MB. By default, there is no maximum row count, and
	// only the byte limit applies.
	MaxResults int64 `json:"maxResults,omitempty"`

	// ParameterMode: Standard SQL only. Set to POSITIONAL to use positional
	// (?) query parameters or to NAMED to use named (@myparam) query
	// parameters in this query.
	ParameterMode string `json:"parameterMode,omitempty"`

	// PreserveNulls: [Deprecated] This property is deprecated.
	PreserveNulls bool `json:"preserveNulls,omitempty"`

	// Query: [Required] A query string, following the BigQuery query
	// syntax, of the query to execute. Example: "SELECT count(f1) FROM
	// [myProjectId:myDatasetId.myTableId]".
	Query string `json:"query,omitempty"`

	// QueryParameters: Query parameters for Standard SQL queries.
	QueryParameters []*QueryParameter `json:"queryParameters,omitempty"`

	// TimeoutMs: [Optional] How long to wait for the query to complete, in
	// milliseconds, before the request times out and returns. Note that
	// this is only a timeout for the request, not the query. If the query
	// takes longer to run than the timeout value, the call returns without
	// any results and with the 'jobComplete' flag set to false. You can
	// call GetQueryResults() to wait for the query to complete and read the
	// results. The default value is 10000 milliseconds (10 seconds).
	TimeoutMs int64 `json:"timeoutMs,omitempty"`

	// UseLegacySql: Specifies whether to use BigQuery's legacy SQL dialect
	// for this query. The default value is true. If set to false, the query
	// will use BigQuery's standard SQL:
	// https://cloud.google.com/bigquery/sql-reference/ When useLegacySql is
	// set to false, the value of flattenResults is ignored; query will be
	// run as if flattenResults is false.
	//
	// Default: true
	UseLegacySql *bool `json:"useLegacySql,omitempty"`

	// UseQueryCache: [Optional] Whether to look for the result in the query
	// cache. The query cache is a best-effort cache that will be flushed
	// whenever tables in the query are modified. The default value is true.
	//
	// Default: true
	UseQueryCache *bool `json:"useQueryCache,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DefaultDataset") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DefaultDataset") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *QueryRequest) MarshalJSON() ([]byte, error) {
	type noMethod QueryRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type QueryResponse struct {
	// CacheHit: Whether the query result was fetched from the query cache.
	CacheHit bool `json:"cacheHit,omitempty"`

	// Errors: [Output-only] The first errors or warnings encountered during
	// the running of the job. The final message includes the number of
	// errors that caused the process to stop. Errors here do not
	// necessarily mean that the job has completed or was unsuccessful.
	Errors []*ErrorProto `json:"errors,omitempty"`

	// JobComplete: Whether the query has completed or not. If rows or
	// totalRows are present, this will always be true. If this is false,
	// totalRows will not be available.
	JobComplete bool `json:"jobComplete,omitempty"`

	// JobReference: Reference to the Job that was created to run the query.
	// This field will be present even if the original request timed out, in
	// which case GetQueryResults can be used to read the results once the
	// query has completed. Since this API only returns the first page of
	// results, subsequent pages can be fetched via the same mechanism
	// (GetQueryResults).
	JobReference *JobReference `json:"jobReference,omitempty"`

	// Kind: The resource type.
	Kind string `json:"kind,omitempty"`

	// NumDmlAffectedRows: [Output-only] The number of rows affected by a
	// DML statement. Present only for DML statements INSERT, UPDATE or
	// DELETE.
	NumDmlAffectedRows int64 `json:"numDmlAffectedRows,omitempty,string"`

	// PageToken: A token used for paging results.
	PageToken string `json:"pageToken,omitempty"`

	// Rows: An object with as many results as can be contained within the
	// maximum permitted reply size. To get any additional rows, you can
	// call GetQueryResults and specify the jobReference returned above.
	Rows []*TableRow `json:"rows,omitempty"`

	// Schema: The schema of the results. Present only when the query
	// completes successfully.
	Schema *TableSchema `json:"schema,omitempty"`

	// TotalBytesProcessed: The total number of bytes processed for this
	// query. If this query was a dry run, this is the number of bytes that
	// would be processed if the query were run.
	TotalBytesProcessed int64 `json:"totalBytesProcessed,omitempty,string"`

	// TotalRows: The total number of rows in the complete query result set,
	// which can be more than the number of rows in this single page of
	// results.
	TotalRows uint64 `json:"totalRows,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CacheHit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CacheHit") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryResponse) MarshalJSON() ([]byte, error) {
	type noMethod QueryResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Streamingbuffer struct {
	// EstimatedBytes: [Output-only] A lower-bound estimate of the number of
	// bytes currently in the streaming buffer.
	EstimatedBytes uint64 `json:"estimatedBytes,omitempty,string"`

	// EstimatedRows: [Output-only] A lower-bound estimate of the number of
	// rows currently in the streaming buffer.
	EstimatedRows uint64 `json:"estimatedRows,omitempty,string"`

	// OldestEntryTime: [Output-only] Contains the timestamp of the oldest
	// entry in the streaming buffer, in milliseconds since the epoch, if
	// the streaming buffer is available.
	OldestEntryTime uint64 `json:"oldestEntryTime,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "EstimatedBytes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EstimatedBytes") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Streamingbuffer) MarshalJSON() ([]byte, error) {
	type noMethod Streamingbuffer
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Table struct {
	// CreationTime: [Output-only] The time when this table was created, in
	// milliseconds since the epoch.
	CreationTime int64 `json:"creationTime,omitempty,string"`

	// Description: [Optional] A user-friendly description of this table.
	Description string `json:"description,omitempty"`

	// EncryptionConfiguration: [Experimental] Custom encryption
	// configuration (e.g., Cloud KMS keys).
	EncryptionConfiguration *EncryptionConfiguration `json:"encryptionConfiguration,omitempty"`

	// Etag: [Output-only] A hash of this resource.
	Etag string `json:"etag,omitempty"`

	// ExpirationTime: [Optional] The time when this table expires, in
	// milliseconds since the epoch. If not present, the table will persist
	// indefinitely. Expired tables will be deleted and their storage
	// reclaimed.
	ExpirationTime int64 `json:"expirationTime,omitempty,string"`

	// ExternalDataConfiguration: [Optional] Describes the data format,
	// location, and other properties of a table stored outside of BigQuery.
	// By defining these properties, the data source can then be queried as
	// if it were a standard BigQuery table.
	ExternalDataConfiguration *ExternalDataConfiguration `json:"externalDataConfiguration,omitempty"`

	// FriendlyName: [Optional] A descriptive name for this table.
	FriendlyName string `json:"friendlyName,omitempty"`

	// Id: [Output-only] An opaque ID uniquely identifying the table.
	Id string `json:"id,omitempty"`

	// Kind: [Output-only] The type of the resource.
	Kind string `json:"kind,omitempty"`

	// Labels: [Experimental] The labels associated with this table. You can
	// use these to organize and group your tables. Label keys and values
	// can be no longer than 63 characters, can only contain lowercase
	// letters, numeric characters, underscores and dashes. International
	// characters are allowed. Label values are optional. Label keys must
	// start with a letter and each label in the list must have a different
	// key.
	Labels map[string]string `json:"labels,omitempty"`

	// LastModifiedTime: [Output-only] The time when this table was last
	// modified, in milliseconds since the epoch.
	LastModifiedTime uint64 `json:"lastModifiedTime,omitempty,string"`

	// Location: [Output-only] The geographic location where the table
	// resides. This value is inherited from the dataset.
	Location string `json:"location,omitempty"`

	// NumBytes: [Output-only] The size of this table in bytes, excluding
	// any data in the streaming buffer.
	NumBytes int64 `json:"numBytes,omitempty,string"`

	// NumLongTermBytes: [Output-only] The number of bytes in the table that
	// are considered "long-term storage".
	NumLongTermBytes int64 `json:"numLongTermBytes,omitempty,string"`

	// NumRows: [Output-only] The number of rows of data in this table,
	// excluding any data in the streaming buffer.
	NumRows uint64 `json:"numRows,omitempty,string"`

	// Schema: [Optional] Describes the schema of this table.
	Schema *TableSchema `json:"schema,omitempty"`

	// SelfLink: [Output-only] A URL that can be used to access this
	// resource again.
	SelfLink string `json:"selfLink,omitempty"`

	// StreamingBuffer: [Output-only] Contains information regarding this
	// table's streaming buffer, if one is present. This field will be
	// absent if the table is not being streamed to or if there is no data
	// in the streaming buffer.
	StreamingBuffer *Streamingbuffer `json:"streamingBuffer,omitempty"`

	// TableReference: [Required] Reference describing the ID of this table.
	TableReference *TableReference `json:"tableReference,omitempty"`

	// TimePartitioning: [Experimental] If specified, configures time-based
	// partitioning for this table.
	TimePartitioning *TimePartitioning `json:"timePartitioning,omitempty"`

	// Type: [Output-only] Describes the table type. The following values
	// are supported: TABLE: A normal BigQuery table. VIEW: A virtual table
	// defined by a SQL query. EXTERNAL: A table that references data stored
	// in an external storage system, such as Google Cloud Storage. The
	// default value is TABLE.
	Type string `json:"type,omitempty"`

	// View: [Optional] The view definition.
	View *ViewDefinition `json:"view,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Table) MarshalJSON() ([]byte, error) {
	type noMethod Table
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableCell struct {
	V interface{} `json:"v,omitempty"`

	// ForceSendFields is a list of field names (e.g. "V") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "V") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableCell) MarshalJSON() ([]byte, error) {
	type noMethod TableCell
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableDataInsertAllRequest struct {
	// IgnoreUnknownValues: [Optional] Accept rows that contain values that
	// do not match the schema. The unknown values are ignored. Default is
	// false, which treats unknown values as errors.
	IgnoreUnknownValues bool `json:"ignoreUnknownValues,omitempty"`

	// Kind: The resource type of the response.
	Kind string `json:"kind,omitempty"`

	// Rows: The rows to insert.
	Rows []*TableDataInsertAllRequestRows `json:"rows,omitempty"`

	// SkipInvalidRows: [Optional] Insert all valid rows of a request, even
	// if invalid rows exist. The default value is false, which causes the
	// entire request to fail if any invalid rows exist.
	SkipInvalidRows bool `json:"skipInvalidRows,omitempty"`

	// TemplateSuffix: [Experimental] If specified, treats the destination
	// table as a base template, and inserts the rows into an instance table
	// named "{destination}{templateSuffix}". BigQuery will manage creation
	// of the instance table, using the schema of the base template table.
	// See
	// https://cloud.google.com/bigquery/streaming-data-into-bigquery#template-tables for considerations when working with templates
	// tables.
	TemplateSuffix string `json:"templateSuffix,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IgnoreUnknownValues")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "IgnoreUnknownValues") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TableDataInsertAllRequest) MarshalJSON() ([]byte, error) {
	type noMethod TableDataInsertAllRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableDataInsertAllRequestRows struct {
	// InsertId: [Optional] A unique ID for each row. BigQuery uses this
	// property to detect duplicate insertion requests on a best-effort
	// basis.
	InsertId string `json:"insertId,omitempty"`

	// Json: [Required] A JSON object that contains a row of data. The
	// object's properties and values must match the destination table's
	// schema.
	Json map[string]JsonValue `json:"json,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InsertId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InsertId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableDataInsertAllRequestRows) MarshalJSON() ([]byte, error) {
	type noMethod TableDataInsertAllRequestRows
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableDataInsertAllResponse struct {
	// InsertErrors: An array of errors for rows that were not inserted.
	InsertErrors []*TableDataInsertAllResponseInsertErrors `json:"insertErrors,omitempty"`

	// Kind: The resource type of the response.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "InsertErrors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InsertErrors") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableDataInsertAllResponse) MarshalJSON() ([]byte, error) {
	type noMethod TableDataInsertAllResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableDataInsertAllResponseInsertErrors struct {
	// Errors: Error information for the row indicated by the index
	// property.
	Errors []*ErrorProto `json:"errors,omitempty"`

	// Index: The index of the row that error applies to.
	Index int64 `json:"index,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Errors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Errors") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableDataInsertAllResponseInsertErrors) MarshalJSON() ([]byte, error) {
	type noMethod TableDataInsertAllResponseInsertErrors
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableDataList struct {
	// Etag: A hash of this page of results.
	Etag string `json:"etag,omitempty"`

	// Kind: The resource type of the response.
	Kind string `json:"kind,omitempty"`

	// PageToken: A token used for paging results. Providing this token
	// instead of the startIndex parameter can help you retrieve stable
	// results when an underlying table is changing.
	PageToken string `json:"pageToken,omitempty"`

	// Rows: Rows of results.
	Rows []*TableRow `json:"rows,omitempty"`

	// TotalRows: The total number of rows in the complete table.
	TotalRows int64 `json:"totalRows,omitempty,string"`

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

func (s *TableDataList) MarshalJSON() ([]byte, error) {
	type noMethod TableDataList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableFieldSchema struct {
	// Description: [Optional] The field description. The maximum length is
	// 1,024 characters.
	Description string `json:"description,omitempty"`

	// Fields: [Optional] Describes the nested schema fields if the type
	// property is set to RECORD.
	Fields []*TableFieldSchema `json:"fields,omitempty"`

	// Mode: [Optional] The field mode. Possible values include NULLABLE,
	// REQUIRED and REPEATED. The default value is NULLABLE.
	Mode string `json:"mode,omitempty"`

	// Name: [Required] The field name. The name must contain only letters
	// (a-z, A-Z), numbers (0-9), or underscores (_), and must start with a
	// letter or underscore. The maximum length is 128 characters.
	Name string `json:"name,omitempty"`

	// Type: [Required] The field data type. Possible values include STRING,
	// BYTES, INTEGER, INT64 (same as INTEGER), FLOAT, FLOAT64 (same as
	// FLOAT), BOOLEAN, BOOL (same as BOOLEAN), TIMESTAMP, DATE, TIME,
	// DATETIME, RECORD (where RECORD indicates that the field contains a
	// nested schema) or STRUCT (same as RECORD).
	Type string `json:"type,omitempty"`

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

func (s *TableFieldSchema) MarshalJSON() ([]byte, error) {
	type noMethod TableFieldSchema
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableList struct {
	// Etag: A hash of this page of results.
	Etag string `json:"etag,omitempty"`

	// Kind: The type of list.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to request the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Tables: Tables in the requested dataset.
	Tables []*TableListTables `json:"tables,omitempty"`

	// TotalItems: The total number of tables in the dataset.
	TotalItems int64 `json:"totalItems,omitempty"`

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

func (s *TableList) MarshalJSON() ([]byte, error) {
	type noMethod TableList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableListTables struct {
	// FriendlyName: The user-friendly name for this table.
	FriendlyName string `json:"friendlyName,omitempty"`

	// Id: An opaque ID of the table
	Id string `json:"id,omitempty"`

	// Kind: The resource type.
	Kind string `json:"kind,omitempty"`

	// Labels: [Experimental] The labels associated with this table. You can
	// use these to organize and group your tables.
	Labels map[string]string `json:"labels,omitempty"`

	// TableReference: A reference uniquely identifying the table.
	TableReference *TableReference `json:"tableReference,omitempty"`

	// TimePartitioning: [Experimental] The time-based partitioning for this
	// table.
	TimePartitioning *TimePartitioning `json:"timePartitioning,omitempty"`

	// Type: The type of table. Possible values are: TABLE, VIEW.
	Type string `json:"type,omitempty"`

	// View: Additional details for a view.
	View *TableListTablesView `json:"view,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FriendlyName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FriendlyName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableListTables) MarshalJSON() ([]byte, error) {
	type noMethod TableListTables
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TableListTablesView: Additional details for a view.
type TableListTablesView struct {
	// UseLegacySql: True if view is defined in legacy SQL dialect, false if
	// in standard SQL.
	UseLegacySql bool `json:"useLegacySql,omitempty"`

	// ForceSendFields is a list of field names (e.g. "UseLegacySql") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "UseLegacySql") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableListTablesView) MarshalJSON() ([]byte, error) {
	type noMethod TableListTablesView
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableReference struct {
	// DatasetId: [Required] The ID of the dataset containing this table.
	DatasetId string `json:"datasetId,omitempty"`

	// ProjectId: [Required] The ID of the project containing this table.
	ProjectId string `json:"projectId,omitempty"`

	// TableId: [Required] The ID of the table. The ID must contain only
	// letters (a-z, A-Z), numbers (0-9), or underscores (_). The maximum
	// length is 1,024 characters.
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

func (s *TableReference) MarshalJSON() ([]byte, error) {
	type noMethod TableReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableRow struct {
	// F: Represents a single row in the result set, consisting of one or
	// more fields.
	F []*TableCell `json:"f,omitempty"`

	// ForceSendFields is a list of field names (e.g. "F") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "F") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableRow) MarshalJSON() ([]byte, error) {
	type noMethod TableRow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TableSchema struct {
	// Fields: Describes the fields in a table.
	Fields []*TableFieldSchema `json:"fields,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TableSchema) MarshalJSON() ([]byte, error) {
	type noMethod TableSchema
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TimePartitioning struct {
	// ExpirationMs: [Optional] Number of milliseconds for which to keep the
	// storage for a partition.
	ExpirationMs int64 `json:"expirationMs,omitempty,string"`

	// Field: [Experimental] [Optional] If not set, the table is partitioned
	// by pseudo column '_PARTITIONTIME'; if set, the table is partitioned
	// by this field. The field must be a top-level TIMESTAMP or DATE field.
	// Its mode must be NULLABLE or REQUIRED.
	Field string `json:"field,omitempty"`

	// Type: [Required] The only type supported is DAY, which will generate
	// one partition per day.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExpirationMs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExpirationMs") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TimePartitioning) MarshalJSON() ([]byte, error) {
	type noMethod TimePartitioning
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type UserDefinedFunctionResource struct {
	// InlineCode: [Pick one] An inline resource that contains code for a
	// user-defined function (UDF). Providing a inline code resource is
	// equivalent to providing a URI for a file containing the same code.
	InlineCode string `json:"inlineCode,omitempty"`

	// ResourceUri: [Pick one] A code resource to load from a Google Cloud
	// Storage URI (gs://bucket/path).
	ResourceUri string `json:"resourceUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InlineCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InlineCode") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UserDefinedFunctionResource) MarshalJSON() ([]byte, error) {
	type noMethod UserDefinedFunctionResource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ViewDefinition struct {
	// Query: [Required] A query that BigQuery executes when the view is
	// referenced.
	Query string `json:"query,omitempty"`

	// UseLegacySql: Specifies whether to use BigQuery's legacy SQL for this
	// view. The default value is true. If set to false, the view will use
	// BigQuery's standard SQL:
	// https://cloud.google.com/bigquery/sql-reference/ Queries and views
	// that reference this view must use the same flag value.
	UseLegacySql bool `json:"useLegacySql,omitempty"`

	// UserDefinedFunctionResources: Describes user-defined function
	// resources used in the query.
	UserDefinedFunctionResources []*UserDefinedFunctionResource `json:"userDefinedFunctionResources,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Query") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Query") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ViewDefinition) MarshalJSON() ([]byte, error) {
	type noMethod ViewDefinition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "bigquery.datasets.delete":

type DatasetsDeleteCall struct {
	s          *Service
	projectId  string
	datasetId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the dataset specified by the datasetId value. Before
// you can delete a dataset, you must delete all its tables, either
// manually or by specifying deleteContents. Immediately after deletion,
// you can create another dataset with the same name.
func (r *DatasetsService) Delete(projectId string, datasetId string) *DatasetsDeleteCall {
	c := &DatasetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	return c
}

// DeleteContents sets the optional parameter "deleteContents": If True,
// delete all the tables in the dataset. If False and the dataset
// contains tables, the request will fail. Default is False
func (c *DatasetsDeleteCall) DeleteContents(deleteContents bool) *DatasetsDeleteCall {
	c.urlParams_.Set("deleteContents", fmt.Sprint(deleteContents))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsDeleteCall) Fields(s ...googleapi.Field) *DatasetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsDeleteCall) Context(ctx context.Context) *DatasetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.datasets.delete" call.
func (c *DatasetsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes the dataset specified by the datasetId value. Before you can delete a dataset, you must delete all its tables, either manually or by specifying deleteContents. Immediately after deletion, you can create another dataset with the same name.",
	//   "httpMethod": "DELETE",
	//   "id": "bigquery.datasets.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of dataset being deleted",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "deleteContents": {
	//       "description": "If True, delete all the tables in the dataset. If False and the dataset contains tables, the request will fail. Default is False",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the dataset being deleted",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.datasets.get":

type DatasetsGetCall struct {
	s            *Service
	projectId    string
	datasetId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns the dataset specified by datasetID.
func (r *DatasetsService) Get(projectId string, datasetId string) *DatasetsGetCall {
	c := &DatasetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsGetCall) Fields(s ...googleapi.Field) *DatasetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatasetsGetCall) IfNoneMatch(entityTag string) *DatasetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsGetCall) Context(ctx context.Context) *DatasetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.datasets.get" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsGetCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
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
	//   "description": "Returns the dataset specified by datasetID.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.datasets.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the requested dataset",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the requested dataset",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// method id "bigquery.datasets.insert":

type DatasetsInsertCall struct {
	s          *Service
	projectId  string
	dataset    *Dataset
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a new empty dataset.
func (r *DatasetsService) Insert(projectId string, dataset *Dataset) *DatasetsInsertCall {
	c := &DatasetsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.dataset = dataset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsInsertCall) Fields(s ...googleapi.Field) *DatasetsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsInsertCall) Context(ctx context.Context) *DatasetsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsInsertCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.datasets.insert" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsInsertCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
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
	//   "description": "Creates a new empty dataset.",
	//   "httpMethod": "POST",
	//   "id": "bigquery.datasets.insert",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Project ID of the new dataset",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.datasets.list":

type DatasetsListCall struct {
	s            *Service
	projectId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all datasets in the specified project to which you have
// been granted the READER dataset role.
func (r *DatasetsService) List(projectId string) *DatasetsListCall {
	c := &DatasetsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	return c
}

// All sets the optional parameter "all": Whether to list all datasets,
// including hidden ones
func (c *DatasetsListCall) All(all bool) *DatasetsListCall {
	c.urlParams_.Set("all", fmt.Sprint(all))
	return c
}

// Filter sets the optional parameter "filter": An expression for
// filtering the results of the request by label. The syntax is
// "labels.<name>[:<value>]". Multiple filters can be ANDed together by
// connecting with a space. Example: "labels.department:receiving
// labels.active". See Filtering datasets using labels for details.
func (c *DatasetsListCall) Filter(filter string) *DatasetsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of results to return
func (c *DatasetsListCall) MaxResults(maxResults int64) *DatasetsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token,
// returned by a previous call, to request the next page of results
func (c *DatasetsListCall) PageToken(pageToken string) *DatasetsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsListCall) Fields(s ...googleapi.Field) *DatasetsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatasetsListCall) IfNoneMatch(entityTag string) *DatasetsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsListCall) Context(ctx context.Context) *DatasetsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.datasets.list" call.
// Exactly one of *DatasetList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DatasetList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatasetsListCall) Do(opts ...googleapi.CallOption) (*DatasetList, error) {
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
	ret := &DatasetList{
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
	//   "description": "Lists all datasets in the specified project to which you have been granted the READER dataset role.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.datasets.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "all": {
	//       "description": "Whether to list all datasets, including hidden ones",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "filter": {
	//       "description": "An expression for filtering the results of the request by label. The syntax is \"labels.\u003cname\u003e[:\u003cvalue\u003e]\". Multiple filters can be ANDed together by connecting with a space. Example: \"labels.department:receiving labels.active\". See Filtering datasets using labels for details.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token, returned by a previous call, to request the next page of results",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the datasets to be listed",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets",
	//   "response": {
	//     "$ref": "DatasetList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *DatasetsListCall) Pages(ctx context.Context, f func(*DatasetList) error) error {
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

// method id "bigquery.datasets.patch":

type DatasetsPatchCall struct {
	s          *Service
	projectId  string
	datasetId  string
	dataset    *Dataset
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates information in an existing dataset. The update method
// replaces the entire dataset resource, whereas the patch method only
// replaces fields that are provided in the submitted dataset resource.
// This method supports patch semantics.
func (r *DatasetsService) Patch(projectId string, datasetId string, dataset *Dataset) *DatasetsPatchCall {
	c := &DatasetsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.dataset = dataset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsPatchCall) Fields(s ...googleapi.Field) *DatasetsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsPatchCall) Context(ctx context.Context) *DatasetsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsPatchCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.datasets.patch" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsPatchCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
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
	//   "description": "Updates information in an existing dataset. The update method replaces the entire dataset resource, whereas the patch method only replaces fields that are provided in the submitted dataset resource. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "bigquery.datasets.patch",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the dataset being updated",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the dataset being updated",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.datasets.update":

type DatasetsUpdateCall struct {
	s          *Service
	projectId  string
	datasetId  string
	dataset    *Dataset
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates information in an existing dataset. The update method
// replaces the entire dataset resource, whereas the patch method only
// replaces fields that are provided in the submitted dataset resource.
func (r *DatasetsService) Update(projectId string, datasetId string, dataset *Dataset) *DatasetsUpdateCall {
	c := &DatasetsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.dataset = dataset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsUpdateCall) Fields(s ...googleapi.Field) *DatasetsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsUpdateCall) Context(ctx context.Context) *DatasetsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsUpdateCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.datasets.update" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsUpdateCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
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
	//   "description": "Updates information in an existing dataset. The update method replaces the entire dataset resource, whereas the patch method only replaces fields that are provided in the submitted dataset resource.",
	//   "httpMethod": "PUT",
	//   "id": "bigquery.datasets.update",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the dataset being updated",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the dataset being updated",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.jobs.cancel":

type JobsCancelCall struct {
	s          *Service
	projectId  string
	jobId      string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Cancel: Requests that a job be cancelled. This call will return
// immediately, and the client will need to poll for the job status to
// see if the cancel completed successfully. Cancelled jobs may still
// incur costs.
func (r *JobsService) Cancel(projectId string, jobId string) *JobsCancelCall {
	c := &JobsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.jobId = jobId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsCancelCall) Fields(s ...googleapi.Field) *JobsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsCancelCall) Context(ctx context.Context) *JobsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *JobsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *JobsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/jobs/{jobId}/cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"jobId":     c.jobId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.jobs.cancel" call.
// Exactly one of *JobCancelResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *JobCancelResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *JobsCancelCall) Do(opts ...googleapi.CallOption) (*JobCancelResponse, error) {
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
	ret := &JobCancelResponse{
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
	//   "description": "Requests that a job be cancelled. This call will return immediately, and the client will need to poll for the job status to see if the cancel completed successfully. Cancelled jobs may still incur costs.",
	//   "httpMethod": "POST",
	//   "id": "bigquery.jobs.cancel",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "[Required] Job ID of the job to cancel",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] Project ID of the job to cancel",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/jobs/{jobId}/cancel",
	//   "response": {
	//     "$ref": "JobCancelResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.jobs.get":

type JobsGetCall struct {
	s            *Service
	projectId    string
	jobId        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns information about a specific job. Job information is
// available for a six month period after creation. Requires that you're
// the person who ran the job, or have the Is Owner project role.
func (r *JobsService) Get(projectId string, jobId string) *JobsGetCall {
	c := &JobsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.jobId = jobId
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

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *JobsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *JobsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"jobId":     c.jobId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.jobs.get" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsGetCall) Do(opts ...googleapi.CallOption) (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Returns information about a specific job. Job information is available for a six month period after creation. Requires that you're the person who ran the job, or have the Is Owner project role.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.jobs.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "[Required] Job ID of the requested job",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] Project ID of the requested job",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// method id "bigquery.jobs.getQueryResults":

type JobsGetQueryResultsCall struct {
	s            *Service
	projectId    string
	jobId        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetQueryResults: Retrieves the results of a query job.
func (r *JobsService) GetQueryResults(projectId string, jobId string) *JobsGetQueryResultsCall {
	c := &JobsGetQueryResultsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.jobId = jobId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to read
func (c *JobsGetQueryResultsCall) MaxResults(maxResults int64) *JobsGetQueryResultsCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token,
// returned by a previous call, to request the next page of results
func (c *JobsGetQueryResultsCall) PageToken(pageToken string) *JobsGetQueryResultsCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// StartIndex sets the optional parameter "startIndex": Zero-based index
// of the starting row
func (c *JobsGetQueryResultsCall) StartIndex(startIndex uint64) *JobsGetQueryResultsCall {
	c.urlParams_.Set("startIndex", fmt.Sprint(startIndex))
	return c
}

// TimeoutMs sets the optional parameter "timeoutMs": How long to wait
// for the query to complete, in milliseconds, before returning. Default
// is 10 seconds. If the timeout passes before the job completes, the
// 'jobComplete' field in the response will be false
func (c *JobsGetQueryResultsCall) TimeoutMs(timeoutMs int64) *JobsGetQueryResultsCall {
	c.urlParams_.Set("timeoutMs", fmt.Sprint(timeoutMs))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsGetQueryResultsCall) Fields(s ...googleapi.Field) *JobsGetQueryResultsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *JobsGetQueryResultsCall) IfNoneMatch(entityTag string) *JobsGetQueryResultsCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsGetQueryResultsCall) Context(ctx context.Context) *JobsGetQueryResultsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *JobsGetQueryResultsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *JobsGetQueryResultsCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/queries/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"jobId":     c.jobId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.jobs.getQueryResults" call.
// Exactly one of *GetQueryResultsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetQueryResultsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *JobsGetQueryResultsCall) Do(opts ...googleapi.CallOption) (*GetQueryResultsResponse, error) {
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
	ret := &GetQueryResultsResponse{
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
	//   "description": "Retrieves the results of a query job.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.jobs.getQueryResults",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "[Required] Job ID of the query job",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to read",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token, returned by a previous call, to request the next page of results",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] Project ID of the query job",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Zero-based index of the starting row",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "timeoutMs": {
	//       "description": "How long to wait for the query to complete, in milliseconds, before returning. Default is 10 seconds. If the timeout passes before the job completes, the 'jobComplete' field in the response will be false",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "projects/{projectId}/queries/{jobId}",
	//   "response": {
	//     "$ref": "GetQueryResultsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *JobsGetQueryResultsCall) Pages(ctx context.Context, f func(*GetQueryResultsResponse) error) error {
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
		if x.PageToken == "" {
			return nil
		}
		c.PageToken(x.PageToken)
	}
}

// method id "bigquery.jobs.insert":

type JobsInsertCall struct {
	s          *Service
	projectId  string
	job        *Job
	urlParams_ gensupport.URLParams
	mediaInfo_ *gensupport.MediaInfo
	ctx_       context.Context
	header_    http.Header
}

// Insert: Starts a new asynchronous job. Requires the Can View project
// role.
func (r *JobsService) Insert(projectId string, job *Job) *JobsInsertCall {
	c := &JobsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.job = job
	return c
}

// Media specifies the media to upload in one or more chunks. The chunk
// size may be controlled by supplying a MediaOption generated by
// googleapi.ChunkSize. The chunk size defaults to
// googleapi.DefaultUploadChunkSize.The Content-Type header used in the
// upload request will be determined by sniffing the contents of r,
// unless a MediaOption generated by googleapi.ContentType is
// supplied.
// At most one of Media and ResumableMedia may be set.
func (c *JobsInsertCall) Media(r io.Reader, options ...googleapi.MediaOption) *JobsInsertCall {
	c.mediaInfo_ = gensupport.NewInfoFromMedia(r, options)
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx.
//
// Deprecated: use Media instead.
//
// At most one of Media and ResumableMedia may be set. mediaType
// identifies the MIME media type of the upload, such as "image/png". If
// mediaType is "", it will be auto-detected. The provided ctx will
// supersede any context previously provided to the Context method.
func (c *JobsInsertCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *JobsInsertCall {
	c.ctx_ = ctx
	c.mediaInfo_ = gensupport.NewInfoFromResumableMedia(r, size, mediaType)
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *JobsInsertCall) ProgressUpdater(pu googleapi.ProgressUpdater) *JobsInsertCall {
	c.mediaInfo_.SetProgressUpdater(pu)
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
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *JobsInsertCall) Context(ctx context.Context) *JobsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *JobsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *JobsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/jobs")
	if c.mediaInfo_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.mediaInfo_.UploadType())
	}
	if body == nil {
		body = new(bytes.Buffer)
		reqHeaders.Set("Content-Type", "application/json")
	}
	body, cleanup := c.mediaInfo_.UploadRequest(reqHeaders, body)
	defer cleanup()
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.jobs.insert" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsInsertCall) Do(opts ...googleapi.CallOption) (*Job, error) {
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
	rx := c.mediaInfo_.ResumableUpload(res.Header.Get("Location"))
	if rx != nil {
		rx.Client = c.s.client
		rx.UserAgent = c.s.userAgent()
		ctx := c.ctx_
		if ctx == nil {
			ctx = context.TODO()
		}
		res, err = rx.Upload(ctx)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
		if err := googleapi.CheckResponse(res); err != nil {
			return nil, err
		}
	}
	ret := &Job{
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
	//   "description": "Starts a new asynchronous job. Requires the Can View project role.",
	//   "httpMethod": "POST",
	//   "id": "bigquery.jobs.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/bigquery/v2/projects/{projectId}/jobs"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/bigquery/v2/projects/{projectId}/jobs"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Project ID of the project that will be billed for the job",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/jobs",
	//   "request": {
	//     "$ref": "Job"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "bigquery.jobs.list":

type JobsListCall struct {
	s            *Service
	projectId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all jobs that you started in the specified project. Job
// information is available for a six month period after creation. The
// job list is sorted in reverse chronological order, by job creation
// time. Requires the Can View project role, or the Is Owner project
// role if you set the allUsers property.
func (r *JobsService) List(projectId string) *JobsListCall {
	c := &JobsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	return c
}

// AllUsers sets the optional parameter "allUsers": Whether to display
// jobs owned by all users in the project. Default false
func (c *JobsListCall) AllUsers(allUsers bool) *JobsListCall {
	c.urlParams_.Set("allUsers", fmt.Sprint(allUsers))
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *JobsListCall) MaxResults(maxResults int64) *JobsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token,
// returned by a previous call, to request the next page of results
func (c *JobsListCall) PageToken(pageToken string) *JobsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Projection sets the optional parameter "projection": Restrict
// information returned to a set of selected fields
//
// Possible values:
//   "full" - Includes all job data
//   "minimal" - Does not include the job configuration
func (c *JobsListCall) Projection(projection string) *JobsListCall {
	c.urlParams_.Set("projection", projection)
	return c
}

// StateFilter sets the optional parameter "stateFilter": Filter for job
// state
//
// Possible values:
//   "done" - Finished jobs
//   "pending" - Pending jobs
//   "running" - Running jobs
func (c *JobsListCall) StateFilter(stateFilter ...string) *JobsListCall {
	c.urlParams_.SetMulti("stateFilter", append([]string{}, stateFilter...))
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

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *JobsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *JobsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/jobs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.jobs.list" call.
// Exactly one of *JobList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *JobList.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *JobsListCall) Do(opts ...googleapi.CallOption) (*JobList, error) {
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
	ret := &JobList{
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
	//   "description": "Lists all jobs that you started in the specified project. Job information is available for a six month period after creation. The job list is sorted in reverse chronological order, by job creation time. Requires the Can View project role, or the Is Owner project role if you set the allUsers property.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.jobs.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "allUsers": {
	//       "description": "Whether to display jobs owned by all users in the project. Default false",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token, returned by a previous call, to request the next page of results",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the jobs to list",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Restrict information returned to a set of selected fields",
	//       "enum": [
	//         "full",
	//         "minimal"
	//       ],
	//       "enumDescriptions": [
	//         "Includes all job data",
	//         "Does not include the job configuration"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "stateFilter": {
	//       "description": "Filter for job state",
	//       "enum": [
	//         "done",
	//         "pending",
	//         "running"
	//       ],
	//       "enumDescriptions": [
	//         "Finished jobs",
	//         "Pending jobs",
	//         "Running jobs"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/jobs",
	//   "response": {
	//     "$ref": "JobList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *JobsListCall) Pages(ctx context.Context, f func(*JobList) error) error {
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

// method id "bigquery.jobs.query":

type JobsQueryCall struct {
	s            *Service
	projectId    string
	queryrequest *QueryRequest
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Query: Runs a BigQuery SQL query synchronously and returns query
// results if the query completes within a specified timeout.
func (r *JobsService) Query(projectId string, queryrequest *QueryRequest) *JobsQueryCall {
	c := &JobsQueryCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.queryrequest = queryrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsQueryCall) Fields(s ...googleapi.Field) *JobsQueryCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsQueryCall) Context(ctx context.Context) *JobsQueryCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *JobsQueryCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *JobsQueryCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.queryrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/queries")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.jobs.query" call.
// Exactly one of *QueryResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *QueryResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *JobsQueryCall) Do(opts ...googleapi.CallOption) (*QueryResponse, error) {
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
	ret := &QueryResponse{
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
	//   "description": "Runs a BigQuery SQL query synchronously and returns query results if the query completes within a specified timeout.",
	//   "httpMethod": "POST",
	//   "id": "bigquery.jobs.query",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Project ID of the project billed for the query",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/queries",
	//   "request": {
	//     "$ref": "QueryRequest"
	//   },
	//   "response": {
	//     "$ref": "QueryResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// method id "bigquery.projects.getServiceAccount":

type ProjectsGetServiceAccountCall struct {
	s            *Service
	projectId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetServiceAccount: Returns the email address of the service account
// for your project used for interactions with Google Cloud KMS.
func (r *ProjectsService) GetServiceAccount(projectId string) *ProjectsGetServiceAccountCall {
	c := &ProjectsGetServiceAccountCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsGetServiceAccountCall) Fields(s ...googleapi.Field) *ProjectsGetServiceAccountCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsGetServiceAccountCall) IfNoneMatch(entityTag string) *ProjectsGetServiceAccountCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsGetServiceAccountCall) Context(ctx context.Context) *ProjectsGetServiceAccountCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsGetServiceAccountCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsGetServiceAccountCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/serviceAccount")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.projects.getServiceAccount" call.
// Exactly one of *GetServiceAccountResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GetServiceAccountResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsGetServiceAccountCall) Do(opts ...googleapi.CallOption) (*GetServiceAccountResponse, error) {
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
	ret := &GetServiceAccountResponse{
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
	//   "description": "Returns the email address of the service account for your project used for interactions with Google Cloud KMS.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.projects.getServiceAccount",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Project ID for which the service account is requested.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/serviceAccount",
	//   "response": {
	//     "$ref": "GetServiceAccountResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// method id "bigquery.projects.list":

type ProjectsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all projects to which you have been granted any project
// role.
func (r *ProjectsService) List() *ProjectsListCall {
	c := &ProjectsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *ProjectsListCall) MaxResults(maxResults int64) *ProjectsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token,
// returned by a previous call, to request the next page of results
func (c *ProjectsListCall) PageToken(pageToken string) *ProjectsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsListCall) Fields(s ...googleapi.Field) *ProjectsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsListCall) IfNoneMatch(entityTag string) *ProjectsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsListCall) Context(ctx context.Context) *ProjectsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.projects.list" call.
// Exactly one of *ProjectList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProjectList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsListCall) Do(opts ...googleapi.CallOption) (*ProjectList, error) {
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
	ret := &ProjectList{
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
	//   "description": "Lists all projects to which you have been granted any project role.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.projects.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token, returned by a previous call, to request the next page of results",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects",
	//   "response": {
	//     "$ref": "ProjectList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsListCall) Pages(ctx context.Context, f func(*ProjectList) error) error {
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

// method id "bigquery.tabledata.insertAll":

type TabledataInsertAllCall struct {
	s                         *Service
	projectId                 string
	datasetId                 string
	tableId                   string
	tabledatainsertallrequest *TableDataInsertAllRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// InsertAll: Streams data into BigQuery one record at a time without
// needing to run a load job. Requires the WRITER dataset role.
func (r *TabledataService) InsertAll(projectId string, datasetId string, tableId string, tabledatainsertallrequest *TableDataInsertAllRequest) *TabledataInsertAllCall {
	c := &TabledataInsertAllCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	c.tabledatainsertallrequest = tabledatainsertallrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TabledataInsertAllCall) Fields(s ...googleapi.Field) *TabledataInsertAllCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TabledataInsertAllCall) Context(ctx context.Context) *TabledataInsertAllCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TabledataInsertAllCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TabledataInsertAllCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tabledatainsertallrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables/{tableId}/insertAll")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
		"tableId":   c.tableId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tabledata.insertAll" call.
// Exactly one of *TableDataInsertAllResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TableDataInsertAllResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TabledataInsertAllCall) Do(opts ...googleapi.CallOption) (*TableDataInsertAllResponse, error) {
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
	ret := &TableDataInsertAllResponse{
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
	//   "description": "Streams data into BigQuery one record at a time without needing to run a load job. Requires the WRITER dataset role.",
	//   "httpMethod": "POST",
	//   "id": "bigquery.tabledata.insertAll",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the destination table.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the destination table.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "description": "Table ID of the destination table.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}/insertAll",
	//   "request": {
	//     "$ref": "TableDataInsertAllRequest"
	//   },
	//   "response": {
	//     "$ref": "TableDataInsertAllResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/bigquery.insertdata",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.tabledata.list":

type TabledataListCall struct {
	s            *Service
	projectId    string
	datasetId    string
	tableId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves table data from a specified set of rows. Requires the
// READER dataset role.
func (r *TabledataService) List(projectId string, datasetId string, tableId string) *TabledataListCall {
	c := &TabledataListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *TabledataListCall) MaxResults(maxResults int64) *TabledataListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token,
// returned by a previous call, identifying the result set
func (c *TabledataListCall) PageToken(pageToken string) *TabledataListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// SelectedFields sets the optional parameter "selectedFields": List of
// fields to return (comma-separated). If unspecified, all fields are
// returned
func (c *TabledataListCall) SelectedFields(selectedFields string) *TabledataListCall {
	c.urlParams_.Set("selectedFields", selectedFields)
	return c
}

// StartIndex sets the optional parameter "startIndex": Zero-based index
// of the starting row to read
func (c *TabledataListCall) StartIndex(startIndex uint64) *TabledataListCall {
	c.urlParams_.Set("startIndex", fmt.Sprint(startIndex))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TabledataListCall) Fields(s ...googleapi.Field) *TabledataListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TabledataListCall) IfNoneMatch(entityTag string) *TabledataListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TabledataListCall) Context(ctx context.Context) *TabledataListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TabledataListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TabledataListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables/{tableId}/data")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
		"tableId":   c.tableId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tabledata.list" call.
// Exactly one of *TableDataList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TableDataList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TabledataListCall) Do(opts ...googleapi.CallOption) (*TableDataList, error) {
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
	ret := &TableDataList{
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
	//   "description": "Retrieves table data from a specified set of rows. Requires the READER dataset role.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.tabledata.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the table to read",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token, returned by a previous call, identifying the result set",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the table to read",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "selectedFields": {
	//       "description": "List of fields to return (comma-separated). If unspecified, all fields are returned",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Zero-based index of the starting row to read",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "description": "Table ID of the table to read",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}/data",
	//   "response": {
	//     "$ref": "TableDataList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TabledataListCall) Pages(ctx context.Context, f func(*TableDataList) error) error {
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
		if x.PageToken == "" {
			return nil
		}
		c.PageToken(x.PageToken)
	}
}

// method id "bigquery.tables.delete":

type TablesDeleteCall struct {
	s          *Service
	projectId  string
	datasetId  string
	tableId    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the table specified by tableId from the dataset. If
// the table contains data, all the data will be deleted.
func (r *TablesService) Delete(projectId string, datasetId string, tableId string) *TablesDeleteCall {
	c := &TablesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TablesDeleteCall) Fields(s ...googleapi.Field) *TablesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TablesDeleteCall) Context(ctx context.Context) *TablesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TablesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TablesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
		"tableId":   c.tableId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tables.delete" call.
func (c *TablesDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes the table specified by tableId from the dataset. If the table contains data, all the data will be deleted.",
	//   "httpMethod": "DELETE",
	//   "id": "bigquery.tables.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the table to delete",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the table to delete",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "description": "Table ID of the table to delete",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.tables.get":

type TablesGetCall struct {
	s            *Service
	projectId    string
	datasetId    string
	tableId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the specified table resource by table ID. This method does
// not return the data in the table, it only returns the table resource,
// which describes the structure of this table.
func (r *TablesService) Get(projectId string, datasetId string, tableId string) *TablesGetCall {
	c := &TablesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	return c
}

// SelectedFields sets the optional parameter "selectedFields": List of
// fields to return (comma-separated). If unspecified, all fields are
// returned
func (c *TablesGetCall) SelectedFields(selectedFields string) *TablesGetCall {
	c.urlParams_.Set("selectedFields", selectedFields)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TablesGetCall) Fields(s ...googleapi.Field) *TablesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TablesGetCall) IfNoneMatch(entityTag string) *TablesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TablesGetCall) Context(ctx context.Context) *TablesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TablesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TablesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
		"tableId":   c.tableId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tables.get" call.
// Exactly one of *Table or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Table.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TablesGetCall) Do(opts ...googleapi.CallOption) (*Table, error) {
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
	ret := &Table{
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
	//   "description": "Gets the specified table resource by table ID. This method does not return the data in the table, it only returns the table resource, which describes the structure of this table.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.tables.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the requested table",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the requested table",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "selectedFields": {
	//       "description": "List of fields to return (comma-separated). If unspecified, all fields are returned",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "description": "Table ID of the requested table",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// method id "bigquery.tables.insert":

type TablesInsertCall struct {
	s          *Service
	projectId  string
	datasetId  string
	table      *Table
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a new, empty table in the dataset.
func (r *TablesService) Insert(projectId string, datasetId string, table *Table) *TablesInsertCall {
	c := &TablesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.table = table
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TablesInsertCall) Fields(s ...googleapi.Field) *TablesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TablesInsertCall) Context(ctx context.Context) *TablesInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TablesInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TablesInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.table)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tables.insert" call.
// Exactly one of *Table or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Table.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TablesInsertCall) Do(opts ...googleapi.CallOption) (*Table, error) {
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
	ret := &Table{
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
	//   "description": "Creates a new, empty table in the dataset.",
	//   "httpMethod": "POST",
	//   "id": "bigquery.tables.insert",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the new table",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the new table",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables",
	//   "request": {
	//     "$ref": "Table"
	//   },
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.tables.list":

type TablesListCall struct {
	s            *Service
	projectId    string
	datasetId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all tables in the specified dataset. Requires the READER
// dataset role.
func (r *TablesService) List(projectId string, datasetId string) *TablesListCall {
	c := &TablesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *TablesListCall) MaxResults(maxResults int64) *TablesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token,
// returned by a previous call, to request the next page of results
func (c *TablesListCall) PageToken(pageToken string) *TablesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TablesListCall) Fields(s ...googleapi.Field) *TablesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TablesListCall) IfNoneMatch(entityTag string) *TablesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TablesListCall) Context(ctx context.Context) *TablesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TablesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TablesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tables.list" call.
// Exactly one of *TableList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TableList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TablesListCall) Do(opts ...googleapi.CallOption) (*TableList, error) {
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
	ret := &TableList{
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
	//   "description": "Lists all tables in the specified dataset. Requires the READER dataset role.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.tables.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the tables to list",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token, returned by a previous call, to request the next page of results",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the tables to list",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables",
	//   "response": {
	//     "$ref": "TableList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TablesListCall) Pages(ctx context.Context, f func(*TableList) error) error {
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

// method id "bigquery.tables.patch":

type TablesPatchCall struct {
	s          *Service
	projectId  string
	datasetId  string
	tableId    string
	table      *Table
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates information in an existing table. The update method
// replaces the entire table resource, whereas the patch method only
// replaces fields that are provided in the submitted table resource.
// This method supports patch semantics.
func (r *TablesService) Patch(projectId string, datasetId string, tableId string, table *Table) *TablesPatchCall {
	c := &TablesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	c.table = table
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TablesPatchCall) Fields(s ...googleapi.Field) *TablesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TablesPatchCall) Context(ctx context.Context) *TablesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TablesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TablesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.table)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
		"tableId":   c.tableId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tables.patch" call.
// Exactly one of *Table or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Table.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TablesPatchCall) Do(opts ...googleapi.CallOption) (*Table, error) {
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
	ret := &Table{
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
	//   "description": "Updates information in an existing table. The update method replaces the entire table resource, whereas the patch method only replaces fields that are provided in the submitted table resource. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "bigquery.tables.patch",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the table to update",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the table to update",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "description": "Table ID of the table to update",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "request": {
	//     "$ref": "Table"
	//   },
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "bigquery.tables.update":

type TablesUpdateCall struct {
	s          *Service
	projectId  string
	datasetId  string
	tableId    string
	table      *Table
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates information in an existing table. The update method
// replaces the entire table resource, whereas the patch method only
// replaces fields that are provided in the submitted table resource.
func (r *TablesService) Update(projectId string, datasetId string, tableId string, table *Table) *TablesUpdateCall {
	c := &TablesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	c.table = table
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TablesUpdateCall) Fields(s ...googleapi.Field) *TablesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TablesUpdateCall) Context(ctx context.Context) *TablesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TablesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TablesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.table)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"datasetId": c.datasetId,
		"tableId":   c.tableId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "bigquery.tables.update" call.
// Exactly one of *Table or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Table.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TablesUpdateCall) Do(opts ...googleapi.CallOption) (*Table, error) {
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
	ret := &Table{
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
	//   "description": "Updates information in an existing table. The update method replaces the entire table resource, whereas the patch method only replaces fields that are provided in the submitted table resource.",
	//   "httpMethod": "PUT",
	//   "id": "bigquery.tables.update",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset ID of the table to update",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID of the table to update",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "description": "Table ID of the table to update",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "request": {
	//     "$ref": "Table"
	//   },
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
