// Package datastore provides access to the Google Cloud Datastore API.
//
// See https://cloud.google.com/datastore/
//
// Usage example:
//
//   import "google.golang.org/api/datastore/v1beta1"
//   ...
//   datastoreService, err := datastore.New(oauthHttpClient)
package datastore // import "google.golang.org/api/datastore/v1beta1"

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

const apiId = "datastore:v1beta1"
const apiName = "datastore"
const apiVersion = "v1beta1"
const basePath = "https://datastore.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View and manage your Google Cloud Datastore data
	DatastoreScope = "https://www.googleapis.com/auth/datastore"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Projects = NewProjectsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Projects *ProjectsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	return rs
}

type ProjectsService struct {
	s *Service
}

// GoogleDatastoreAdminV1beta1CommonMetadata: Metadata common to all
// Datastore Admin operations.
type GoogleDatastoreAdminV1beta1CommonMetadata struct {
	// EndTime: The time the operation ended, either successfully or
	// otherwise.
	EndTime string `json:"endTime,omitempty"`

	// Labels: The client-assigned labels which were provided when the
	// operation was
	// created.  May also include additional labels.
	Labels map[string]string `json:"labels,omitempty"`

	// OperationType: The type of the operation.  Can be used as a filter
	// in
	// ListOperationsRequest.
	//
	// Possible values:
	//   "OPERATION_TYPE_UNSPECIFIED" - Unspecified.
	//   "EXPORT_ENTITIES" - ExportEntities.
	//   "IMPORT_ENTITIES" - ImportEntities.
	//   "BUILD_INDEX" - Build an index.
	//   "CLEAR_INDEX" - Clear an index.
	OperationType string `json:"operationType,omitempty"`

	// StartTime: The time that work began on the operation.
	StartTime string `json:"startTime,omitempty"`

	// State: The current state of the Operation.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED" - Unspecified.
	//   "INITIALIZING" - Request is being prepared for processing.
	//   "PROCESSING" - Request is actively being processed.
	//   "CANCELLING" - Request is in the process of being cancelled after
	// user called
	// longrunning.Operations.CancelOperation on the operation.
	//   "FINALIZING" - Request has been processed and is in its
	// finalization stage.
	//   "SUCCESSFUL" - Request has completed successfully.
	//   "FAILED" - Request has finished being processed, but encountered an
	// error.
	//   "CANCELLED" - Request has finished being cancelled after user
	// called
	// longrunning.Operations.CancelOperation.
	State string `json:"state,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1CommonMetadata) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1CommonMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleDatastoreAdminV1beta1EntityFilter: Identifies a subset of
// entities in a project.  This is specified as
// combinations of kind + namespace (either or both of which may be all,
// as
// described in the following examples).
// Example usage:
//
// Entire project:
//   kinds=[], namespace_ids=[]
//
// Kinds Foo and Bar in all namespaces:
//   kinds=['Foo', 'Bar'], namespace_ids=[]
//
// Kinds Foo and Bar only in the default namespace:
//   kinds=['Foo', 'Bar'], namespace_ids=['']
//
// Kinds Foo and Bar in both the default and Baz namespaces:
//   kinds=['Foo', 'Bar'], namespace_ids=['', 'Baz']
//
// The entire Baz namespace:
//   kinds=[], namespace_ids=['Baz']
type GoogleDatastoreAdminV1beta1EntityFilter struct {
	// Kinds: If empty, then this represents all kinds.
	Kinds []string `json:"kinds,omitempty"`

	// NamespaceIds: An empty list represents all namespaces.  This is the
	// preferred
	// usage for projects that don't use namespaces.
	//
	// An empty string element represents the default namespace.  This
	// should be
	// used if the project has data in non-default namespaces, but doesn't
	// want to
	// include them.
	// Each namespace in this list must be unique.
	NamespaceIds []string `json:"namespaceIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kinds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Kinds") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1EntityFilter) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1EntityFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleDatastoreAdminV1beta1ExportEntitiesMetadata: Metadata for
// ExportEntities operations.
type GoogleDatastoreAdminV1beta1ExportEntitiesMetadata struct {
	// Common: Metadata common to all Datastore Admin operations.
	Common *GoogleDatastoreAdminV1beta1CommonMetadata `json:"common,omitempty"`

	// EntityFilter: Description of which entities are being exported.
	EntityFilter *GoogleDatastoreAdminV1beta1EntityFilter `json:"entityFilter,omitempty"`

	// OutputUrlPrefix: Location for the export metadata and data files.
	// This will be the same
	// value as
	// the
	// google.datastore.admin.v1beta1.ExportEntitiesRequest.output_url_pr
	// efix
	// field. The final output location is provided
	// in
	// google.datastore.admin.v1beta1.ExportEntitiesResponse.output_url.
	OutputUrlPrefix string `json:"outputUrlPrefix,omitempty"`

	// ProgressBytes: An estimate of the number of bytes processed.
	ProgressBytes *GoogleDatastoreAdminV1beta1Progress `json:"progressBytes,omitempty"`

	// ProgressEntities: An estimate of the number of entities processed.
	ProgressEntities *GoogleDatastoreAdminV1beta1Progress `json:"progressEntities,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Common") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Common") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1ExportEntitiesMetadata) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1ExportEntitiesMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleDatastoreAdminV1beta1ExportEntitiesRequest: The request
// for
// google.datastore.admin.v1beta1.DatastoreAdmin.ExportEntities.
type GoogleDatastoreAdminV1beta1ExportEntitiesRequest struct {
	// EntityFilter: Description of what data from the project is included
	// in the export.
	EntityFilter *GoogleDatastoreAdminV1beta1EntityFilter `json:"entityFilter,omitempty"`

	// Labels: Client-assigned labels.
	Labels map[string]string `json:"labels,omitempty"`

	// OutputUrlPrefix: Location for the export metadata and data
	// files.
	//
	// The full resource URL of the external storage location. Currently,
	// only
	// Google Cloud Storage is supported. So output_url_prefix should be of
	// the
	// form: `gs://BUCKET_NAME[/NAMESPACE_PATH]`, where `BUCKET_NAME` is
	// the
	// name of the Cloud Storage bucket and `NAMESPACE_PATH` is an optional
	// Cloud
	// Storage namespace path (this is not a Cloud Datastore namespace). For
	// more
	// information about Cloud Storage namespace paths, see
	// [Object
	// name
	// considerations](https://cloud.google.com/storage/docs/naming#obje
	// ct-considerations).
	//
	// The resulting files will be nested deeper than the specified URL
	// prefix.
	// The final output URL will be provided in
	// the
	// google.datastore.admin.v1beta1.ExportEntitiesResponse.output_url
	// f
	// ield. That value should be used for subsequent ImportEntities
	// operations.
	//
	// By nesting the data files deeper, the same Cloud Storage bucket can
	// be used
	// in multiple ExportEntities operations without conflict.
	OutputUrlPrefix string `json:"outputUrlPrefix,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EntityFilter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EntityFilter") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1ExportEntitiesRequest) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1ExportEntitiesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleDatastoreAdminV1beta1ExportEntitiesResponse: The response
// for
// google.datastore.admin.v1beta1.DatastoreAdmin.ExportEntities.
type GoogleDatastoreAdminV1beta1ExportEntitiesResponse struct {
	// OutputUrl: Location of the output metadata file. This can be used to
	// begin an import
	// into Cloud Datastore (this project or another project).
	// See
	// google.datastore.admin.v1beta1.ImportEntitiesRequest.input_url.
	// On
	// ly present if the operation completed successfully.
	OutputUrl string `json:"outputUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OutputUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OutputUrl") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1ExportEntitiesResponse) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1ExportEntitiesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleDatastoreAdminV1beta1ImportEntitiesMetadata: Metadata for
// ImportEntities operations.
type GoogleDatastoreAdminV1beta1ImportEntitiesMetadata struct {
	// Common: Metadata common to all Datastore Admin operations.
	Common *GoogleDatastoreAdminV1beta1CommonMetadata `json:"common,omitempty"`

	// EntityFilter: Description of which entities are being imported.
	EntityFilter *GoogleDatastoreAdminV1beta1EntityFilter `json:"entityFilter,omitempty"`

	// InputUrl: The location of the import metadata file. This will be the
	// same value as
	// the
	// google.datastore.admin.v1beta1.ExportEntitiesResponse.output_url
	// field
	// .
	InputUrl string `json:"inputUrl,omitempty"`

	// ProgressBytes: An estimate of the number of bytes processed.
	ProgressBytes *GoogleDatastoreAdminV1beta1Progress `json:"progressBytes,omitempty"`

	// ProgressEntities: An estimate of the number of entities processed.
	ProgressEntities *GoogleDatastoreAdminV1beta1Progress `json:"progressEntities,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Common") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Common") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1ImportEntitiesMetadata) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1ImportEntitiesMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleDatastoreAdminV1beta1ImportEntitiesRequest: The request
// for
// google.datastore.admin.v1beta1.DatastoreAdmin.ImportEntities.
type GoogleDatastoreAdminV1beta1ImportEntitiesRequest struct {
	// EntityFilter: Optionally specify which kinds/namespaces are to be
	// imported. If provided,
	// the list must be a subset of the EntityFilter used in creating the
	// export,
	// otherwise a FAILED_PRECONDITION error will be returned. If no filter
	// is
	// specified then all entities from the export are imported.
	EntityFilter *GoogleDatastoreAdminV1beta1EntityFilter `json:"entityFilter,omitempty"`

	// InputUrl: The full resource URL of the external storage location.
	// Currently, only
	// Google Cloud Storage is supported. So input_url should be of the
	// form:
	// `gs://BUCKET_NAME[/NAMESPACE_PATH]/OVERALL_EXPORT_METADATA_FILE`
	// , where
	// `BUCKET_NAME` is the name of the Cloud Storage bucket,
	// `NAMESPACE_PATH` is
	// an optional Cloud Storage namespace path (this is not a Cloud
	// Datastore
	// namespace), and `OVERALL_EXPORT_METADATA_FILE` is the metadata file
	// written
	// by the ExportEntities operation. For more information about Cloud
	// Storage
	// namespace paths, see
	// [Object
	// name
	// considerations](https://cloud.google.com/storage/docs/naming#obje
	// ct-considerations).
	//
	// For more information,
	// see
	// google.datastore.admin.v1beta1.ExportEntitiesResponse.output_url.
	InputUrl string `json:"inputUrl,omitempty"`

	// Labels: Client-assigned labels.
	Labels map[string]string `json:"labels,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EntityFilter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EntityFilter") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1ImportEntitiesRequest) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1ImportEntitiesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleDatastoreAdminV1beta1Progress: Measures the progress of a
// particular metric.
type GoogleDatastoreAdminV1beta1Progress struct {
	// WorkCompleted: Note that this may be greater than work_estimated.
	WorkCompleted int64 `json:"workCompleted,omitempty,string"`

	// WorkEstimated: An estimate of how much work needs to be performed.
	// May be zero if the
	// work estimate is unavailable.
	WorkEstimated int64 `json:"workEstimated,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "WorkCompleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "WorkCompleted") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleDatastoreAdminV1beta1Progress) MarshalJSON() ([]byte, error) {
	type noMethod GoogleDatastoreAdminV1beta1Progress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleLongrunningOperation: This resource represents a long-running
// operation that is the result of a
// network API call.
type GoogleLongrunningOperation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress.
	// If true, the operation is completed, and either `error` or `response`
	// is
	// available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure or
	// cancellation.
	Error *Status `json:"error,omitempty"`

	// Metadata: Service-specific metadata associated with the operation.
	// It typically
	// contains progress information and common metadata such as create
	// time.
	// Some services might not provide such metadata.  Any method that
	// returns a
	// long-running operation should document the metadata type, if any.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that
	// originally returns it. If you use the default HTTP mapping,
	// the
	// `name` should have the format of `operations/some/unique/name`.
	Name string `json:"name,omitempty"`

	// Response: The normal response of the operation in case of success.
	// If the original
	// method returns no data on success, such as `Delete`, the response
	// is
	// `google.protobuf.Empty`.  If the original method is
	// standard
	// `Get`/`Create`/`Update`, the response should be the resource.  For
	// other
	// methods, the response should have the type `XxxResponse`, where
	// `Xxx`
	// is the original method name.  For example, if the original method
	// name
	// is `TakeSnapshot()`, the inferred response type
	// is
	// `TakeSnapshotResponse`.
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

// Status: The `Status` type defines a logical error model that is
// suitable for different
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
type Status struct {
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

func (s *Status) MarshalJSON() ([]byte, error) {
	type noMethod Status
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "datastore.projects.export":

type ProjectsExportCall struct {
	s                                                *Service
	projectId                                        string
	googledatastoreadminv1beta1exportentitiesrequest *GoogleDatastoreAdminV1beta1ExportEntitiesRequest
	urlParams_                                       gensupport.URLParams
	ctx_                                             context.Context
	header_                                          http.Header
}

// Export: Exports a copy of all or a subset of entities from Google
// Cloud Datastore
// to another storage system, such as Google Cloud Storage. Recent
// updates to
// entities may not be reflected in the export. The export occurs in
// the
// background and its progress can be monitored and managed via
// the
// Operation resource that is created.  The output of an export may only
// be
// used once the associated operation is done. If an export operation
// is
// cancelled before completion it may leave partial data behind in
// Google
// Cloud Storage.
func (r *ProjectsService) Export(projectId string, googledatastoreadminv1beta1exportentitiesrequest *GoogleDatastoreAdminV1beta1ExportEntitiesRequest) *ProjectsExportCall {
	c := &ProjectsExportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.googledatastoreadminv1beta1exportentitiesrequest = googledatastoreadminv1beta1exportentitiesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsExportCall) Fields(s ...googleapi.Field) *ProjectsExportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsExportCall) Context(ctx context.Context) *ProjectsExportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsExportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsExportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googledatastoreadminv1beta1exportentitiesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/projects/{projectId}:export")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.export" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsExportCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	//   "description": "Exports a copy of all or a subset of entities from Google Cloud Datastore\nto another storage system, such as Google Cloud Storage. Recent updates to\nentities may not be reflected in the export. The export occurs in the\nbackground and its progress can be monitored and managed via the\nOperation resource that is created.  The output of an export may only be\nused once the associated operation is done. If an export operation is\ncancelled before completion it may leave partial data behind in Google\nCloud Storage.",
	//   "flatPath": "v1beta1/projects/{projectId}:export",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.export",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Project ID against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/projects/{projectId}:export",
	//   "request": {
	//     "$ref": "GoogleDatastoreAdminV1beta1ExportEntitiesRequest"
	//   },
	//   "response": {
	//     "$ref": "GoogleLongrunningOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.import":

type ProjectsImportCall struct {
	s                                                *Service
	projectId                                        string
	googledatastoreadminv1beta1importentitiesrequest *GoogleDatastoreAdminV1beta1ImportEntitiesRequest
	urlParams_                                       gensupport.URLParams
	ctx_                                             context.Context
	header_                                          http.Header
}

// Import: Imports entities into Google Cloud Datastore. Existing
// entities with the
// same key are overwritten. The import occurs in the background and
// its
// progress can be monitored and managed via the Operation resource that
// is
// created.  If an ImportEntities operation is cancelled, it is
// possible
// that a subset of the data has already been imported to Cloud
// Datastore.
func (r *ProjectsService) Import(projectId string, googledatastoreadminv1beta1importentitiesrequest *GoogleDatastoreAdminV1beta1ImportEntitiesRequest) *ProjectsImportCall {
	c := &ProjectsImportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.googledatastoreadminv1beta1importentitiesrequest = googledatastoreadminv1beta1importentitiesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsImportCall) Fields(s ...googleapi.Field) *ProjectsImportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsImportCall) Context(ctx context.Context) *ProjectsImportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsImportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsImportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googledatastoreadminv1beta1importentitiesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/projects/{projectId}:import")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.import" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsImportCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	//   "description": "Imports entities into Google Cloud Datastore. Existing entities with the\nsame key are overwritten. The import occurs in the background and its\nprogress can be monitored and managed via the Operation resource that is\ncreated.  If an ImportEntities operation is cancelled, it is possible\nthat a subset of the data has already been imported to Cloud Datastore.",
	//   "flatPath": "v1beta1/projects/{projectId}:import",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.import",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Project ID against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/projects/{projectId}:import",
	//   "request": {
	//     "$ref": "GoogleDatastoreAdminV1beta1ImportEntitiesRequest"
	//   },
	//   "response": {
	//     "$ref": "GoogleLongrunningOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}
