// Package datastore provides access to the Google Cloud Datastore API.
//
// See https://cloud.google.com/datastore/
//
// Usage example:
//
//   import "google.golang.org/api/datastore/v1"
//   ...
//   datastoreService, err := datastore.New(oauthHttpClient)
package datastore // import "google.golang.org/api/datastore/v1"

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

const apiId = "datastore:v1"
const apiName = "datastore"
const apiVersion = "v1"
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
	rs.Operations = NewProjectsOperationsService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Operations *ProjectsOperationsService
}

func NewProjectsOperationsService(s *Service) *ProjectsOperationsService {
	rs := &ProjectsOperationsService{s: s}
	return rs
}

type ProjectsOperationsService struct {
	s *Service
}

// AllocateIdsRequest: The request for Datastore.AllocateIds.
type AllocateIdsRequest struct {
	// Keys: A list of keys with incomplete key paths for which to allocate
	// IDs.
	// No key may be reserved/read-only.
	Keys []*Key `json:"keys,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Keys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Keys") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AllocateIdsRequest) MarshalJSON() ([]byte, error) {
	type noMethod AllocateIdsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AllocateIdsResponse: The response for Datastore.AllocateIds.
type AllocateIdsResponse struct {
	// Keys: The keys specified in the request (in the same order), each
	// with
	// its key path completed with a newly allocated ID.
	Keys []*Key `json:"keys,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Keys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Keys") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AllocateIdsResponse) MarshalJSON() ([]byte, error) {
	type noMethod AllocateIdsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ArrayValue: An array value.
type ArrayValue struct {
	// Values: Values in the array.
	// The order of this array may not be preserved if it contains a mix
	// of
	// indexed and unindexed values.
	Values []*Value `json:"values,omitempty"`

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

func (s *ArrayValue) MarshalJSON() ([]byte, error) {
	type noMethod ArrayValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BeginTransactionRequest: The request for Datastore.BeginTransaction.
type BeginTransactionRequest struct {
	// TransactionOptions: Options for a new transaction.
	TransactionOptions *TransactionOptions `json:"transactionOptions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TransactionOptions")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TransactionOptions") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BeginTransactionRequest) MarshalJSON() ([]byte, error) {
	type noMethod BeginTransactionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BeginTransactionResponse: The response for
// Datastore.BeginTransaction.
type BeginTransactionResponse struct {
	// Transaction: The transaction identifier (always present).
	Transaction string `json:"transaction,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Transaction") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Transaction") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BeginTransactionResponse) MarshalJSON() ([]byte, error) {
	type noMethod BeginTransactionResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommitRequest: The request for Datastore.Commit.
type CommitRequest struct {
	// Mode: The type of commit to perform. Defaults to `TRANSACTIONAL`.
	//
	// Possible values:
	//   "MODE_UNSPECIFIED" - Unspecified. This value must not be used.
	//   "TRANSACTIONAL" - Transactional: The mutations are either all
	// applied, or none are applied.
	// Learn about transactions
	// [here](https://cloud.google.com/datastore/docs/concepts/transactions).
	//   "NON_TRANSACTIONAL" - Non-transactional: The mutations may not
	// apply as all or none.
	Mode string `json:"mode,omitempty"`

	// Mutations: The mutations to perform.
	//
	// When mode is `TRANSACTIONAL`, mutations affecting a single entity
	// are
	// applied in order. The following sequences of mutations affecting a
	// single
	// entity are not permitted in a single `Commit` request:
	//
	// - `insert` followed by `insert`
	// - `update` followed by `insert`
	// - `upsert` followed by `insert`
	// - `delete` followed by `update`
	//
	// When mode is `NON_TRANSACTIONAL`, no two mutations may affect a
	// single
	// entity.
	Mutations []*Mutation `json:"mutations,omitempty"`

	// Transaction: The identifier of the transaction associated with the
	// commit. A
	// transaction identifier is returned by a call
	// to
	// Datastore.BeginTransaction.
	Transaction string `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Mode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Mode") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommitRequest) MarshalJSON() ([]byte, error) {
	type noMethod CommitRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommitResponse: The response for Datastore.Commit.
type CommitResponse struct {
	// IndexUpdates: The number of index entries updated during the commit,
	// or zero if none were
	// updated.
	IndexUpdates int64 `json:"indexUpdates,omitempty"`

	// MutationResults: The result of performing the mutations.
	// The i-th mutation result corresponds to the i-th mutation in the
	// request.
	MutationResults []*MutationResult `json:"mutationResults,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "IndexUpdates") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "IndexUpdates") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommitResponse) MarshalJSON() ([]byte, error) {
	type noMethod CommitResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CompositeFilter: A filter that merges multiple other filters using
// the given operator.
type CompositeFilter struct {
	// Filters: The list of filters to combine.
	// Must contain at least one filter.
	Filters []*Filter `json:"filters,omitempty"`

	// Op: The operator for combining multiple filters.
	//
	// Possible values:
	//   "OPERATOR_UNSPECIFIED" - Unspecified. This value must not be used.
	//   "AND" - The results are required to satisfy each of the combined
	// filters.
	Op string `json:"op,omitempty"`

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

func (s *CompositeFilter) MarshalJSON() ([]byte, error) {
	type noMethod CompositeFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated
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
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// Entity: A Datastore data object.
//
// An entity is limited to 1 megabyte when stored. That
// _roughly_
// corresponds to a limit of 1 megabyte for the serialized form of
// this
// message.
type Entity struct {
	// Key: The entity's key.
	//
	// An entity must have a key, unless otherwise documented (for
	// example,
	// an entity in `Value.entity_value` may have no key).
	// An entity's kind is its key path's last element's kind,
	// or null if it has no key.
	Key *Key `json:"key,omitempty"`

	// Properties: The entity's properties.
	// The map's keys are property names.
	// A property name matching regex `__.*__` is reserved.
	// A reserved property name is forbidden in certain documented
	// contexts.
	// The name must not contain more than 500 characters.
	// The name cannot be "".
	Properties map[string]Value `json:"properties,omitempty"`

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

func (s *Entity) MarshalJSON() ([]byte, error) {
	type noMethod Entity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EntityResult: The result of fetching an entity from Datastore.
type EntityResult struct {
	// Cursor: A cursor that points to the position after the result
	// entity.
	// Set only when the `EntityResult` is part of a `QueryResultBatch`
	// message.
	Cursor string `json:"cursor,omitempty"`

	// Entity: The resulting entity.
	Entity *Entity `json:"entity,omitempty"`

	// Version: The version of the entity, a strictly positive number that
	// monotonically
	// increases with changes to the entity.
	//
	// This field is set for `FULL` entity
	// results.
	//
	// For missing entities in `LookupResponse`, this
	// is the version of the snapshot that was used to look up the entity,
	// and it
	// is always set except for eventually consistent reads.
	Version int64 `json:"version,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Cursor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cursor") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EntityResult) MarshalJSON() ([]byte, error) {
	type noMethod EntityResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Filter: A holder for any type of filter.
type Filter struct {
	// CompositeFilter: A composite filter.
	CompositeFilter *CompositeFilter `json:"compositeFilter,omitempty"`

	// PropertyFilter: A filter on a property.
	PropertyFilter *PropertyFilter `json:"propertyFilter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CompositeFilter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompositeFilter") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Filter) MarshalJSON() ([]byte, error) {
	type noMethod Filter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

// GqlQuery: A [GQL
// query](https://cloud.google.com/datastore/docs/apis/gql/gql_reference)
// .
type GqlQuery struct {
	// AllowLiterals: When false, the query string must not contain any
	// literals and instead must
	// bind all values. For example,
	// `SELECT * FROM Kind WHERE a = 'string literal'` is not allowed,
	// while
	// `SELECT * FROM Kind WHERE a = @value` is.
	AllowLiterals bool `json:"allowLiterals,omitempty"`

	// NamedBindings: For each non-reserved named binding site in the query
	// string, there must be
	// a named parameter with that name, but not necessarily the
	// inverse.
	//
	// Key must match regex `A-Za-z_$*`, must not match regex
	// `__.*__`, and must not be "".
	NamedBindings map[string]GqlQueryParameter `json:"namedBindings,omitempty"`

	// PositionalBindings: Numbered binding site @1 references the first
	// numbered parameter,
	// effectively using 1-based indexing, rather than the usual 0.
	//
	// For each binding site numbered i in `query_string`, there must be an
	// i-th
	// numbered parameter. The inverse must also be true.
	PositionalBindings []*GqlQueryParameter `json:"positionalBindings,omitempty"`

	// QueryString: A string of the format
	// described
	// [here](https://cloud.google.com/datastore/docs/apis/gql/gql_
	// reference).
	QueryString string `json:"queryString,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AllowLiterals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowLiterals") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GqlQuery) MarshalJSON() ([]byte, error) {
	type noMethod GqlQuery
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GqlQueryParameter: A binding parameter for a GQL query.
type GqlQueryParameter struct {
	// Cursor: A query cursor. Query cursors are returned in query
	// result batches.
	Cursor string `json:"cursor,omitempty"`

	// Value: A value parameter.
	Value *Value `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cursor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cursor") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GqlQueryParameter) MarshalJSON() ([]byte, error) {
	type noMethod GqlQueryParameter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Key: A unique identifier for an entity.
// If a key's partition ID or any of its path kinds or names
// are
// reserved/read-only, the key is reserved/read-only.
// A reserved/read-only key is forbidden in certain documented contexts.
type Key struct {
	// PartitionId: Entities are partitioned into subsets, currently
	// identified by a project
	// ID and namespace ID.
	// Queries are scoped to a single partition.
	PartitionId *PartitionId `json:"partitionId,omitempty"`

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
	// An entity path is always fully complete: *all* of the entity's
	// ancestors
	// are required to be in the path along with the entity identifier
	// itself.
	// The only exception is that in some documented cases, the identifier
	// in the
	// last path element (for the entity) itself may be omitted. For
	// example,
	// the last path element of the key of `Mutation.insert` may have
	// no
	// identifier.
	//
	// A path can never be empty, and a path can have at most 100 elements.
	Path []*PathElement `json:"path,omitempty"`

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

func (s *Key) MarshalJSON() ([]byte, error) {
	type noMethod Key
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// KindExpression: A representation of a kind.
type KindExpression struct {
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

func (s *KindExpression) MarshalJSON() ([]byte, error) {
	type noMethod KindExpression
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LatLng: An object representing a latitude/longitude pair. This is
// expressed as a pair
// of doubles representing degrees latitude and degrees longitude.
// Unless
// specified otherwise, this must conform to the
// <a
// href="http://www.unoosa.org/pdf/icg/2012/template/WGS_84.pdf">WGS84
// st
// andard</a>. Values must be within normalized ranges.
//
// Example of normalization code in Python:
//
//     def NormalizeLongitude(longitude):
//       """Wraps decimal degrees longitude to [-180.0, 180.0]."""
//       q, r = divmod(longitude, 360.0)
//       if r > 180.0 or (r == 180.0 and q <= -1.0):
//         return r - 360.0
//       return r
//
//     def NormalizeLatLng(latitude, longitude):
//       """Wraps decimal degrees latitude and longitude to
//       [-90.0, 90.0] and [-180.0, 180.0], respectively."""
//       r = latitude % 360.0
//       if r <= 90.0:
//         return r, NormalizeLongitude(longitude)
//       elif r >= 270.0:
//         return r - 360, NormalizeLongitude(longitude)
//       else:
//         return 180 - r, NormalizeLongitude(longitude + 180.0)
//
//     assert 180.0 == NormalizeLongitude(180.0)
//     assert -180.0 == NormalizeLongitude(-180.0)
//     assert -179.0 == NormalizeLongitude(181.0)
//     assert (0.0, 0.0) == NormalizeLatLng(360.0, 0.0)
//     assert (0.0, 0.0) == NormalizeLatLng(-360.0, 0.0)
//     assert (85.0, 180.0) == NormalizeLatLng(95.0, 0.0)
//     assert (-85.0, -170.0) == NormalizeLatLng(-95.0, 10.0)
//     assert (90.0, 10.0) == NormalizeLatLng(90.0, 10.0)
//     assert (-90.0, -10.0) == NormalizeLatLng(-90.0, -10.0)
//     assert (0.0, -170.0) == NormalizeLatLng(-180.0, 10.0)
//     assert (0.0, -170.0) == NormalizeLatLng(180.0, 10.0)
//     assert (-90.0, 10.0) == NormalizeLatLng(270.0, 10.0)
//     assert (90.0, 10.0) == NormalizeLatLng(-270.0, 10.0)
type LatLng struct {
	// Latitude: The latitude in degrees. It must be in the range [-90.0,
	// +90.0].
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: The longitude in degrees. It must be in the range [-180.0,
	// +180.0].
	Longitude float64 `json:"longitude,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Latitude") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Latitude") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LatLng) MarshalJSON() ([]byte, error) {
	type noMethod LatLng
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *LatLng) UnmarshalJSON(data []byte) error {
	type noMethod LatLng
	var s1 struct {
		Latitude  gensupport.JSONFloat64 `json:"latitude"`
		Longitude gensupport.JSONFloat64 `json:"longitude"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Latitude = float64(s1.Latitude)
	s.Longitude = float64(s1.Longitude)
	return nil
}

// LookupRequest: The request for Datastore.Lookup.
type LookupRequest struct {
	// Keys: Keys of entities to look up.
	Keys []*Key `json:"keys,omitempty"`

	// ReadOptions: The options for this lookup request.
	ReadOptions *ReadOptions `json:"readOptions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Keys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Keys") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LookupRequest) MarshalJSON() ([]byte, error) {
	type noMethod LookupRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LookupResponse: The response for Datastore.Lookup.
type LookupResponse struct {
	// Deferred: A list of keys that were not looked up due to resource
	// constraints. The
	// order of results in this field is undefined and has no relation to
	// the
	// order of the keys in the input.
	Deferred []*Key `json:"deferred,omitempty"`

	// Found: Entities found as `ResultType.FULL` entities. The order of
	// results in this
	// field is undefined and has no relation to the order of the keys in
	// the
	// input.
	Found []*EntityResult `json:"found,omitempty"`

	// Missing: Entities not found as `ResultType.KEY_ONLY` entities. The
	// order of results
	// in this field is undefined and has no relation to the order of the
	// keys
	// in the input.
	Missing []*EntityResult `json:"missing,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deferred") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Deferred") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LookupResponse) MarshalJSON() ([]byte, error) {
	type noMethod LookupResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Mutation: A mutation to apply to an entity.
type Mutation struct {
	// BaseVersion: The version of the entity that this mutation is being
	// applied to. If this
	// does not match the current version on the server, the mutation
	// conflicts.
	BaseVersion int64 `json:"baseVersion,omitempty,string"`

	// Delete: The key of the entity to delete. The entity may or may not
	// already exist.
	// Must have a complete key path and must not be reserved/read-only.
	Delete *Key `json:"delete,omitempty"`

	// Insert: The entity to insert. The entity must not already exist.
	// The entity key's final path element may be incomplete.
	Insert *Entity `json:"insert,omitempty"`

	// Update: The entity to update. The entity must already exist.
	// Must have a complete key path.
	Update *Entity `json:"update,omitempty"`

	// Upsert: The entity to upsert. The entity may or may not already
	// exist.
	// The entity key's final path element may be incomplete.
	Upsert *Entity `json:"upsert,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BaseVersion") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BaseVersion") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Mutation) MarshalJSON() ([]byte, error) {
	type noMethod Mutation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MutationResult: The result of applying a mutation.
type MutationResult struct {
	// ConflictDetected: Whether a conflict was detected for this mutation.
	// Always false when a
	// conflict detection strategy field is not set in the mutation.
	ConflictDetected bool `json:"conflictDetected,omitempty"`

	// Key: The automatically allocated key.
	// Set only when the mutation allocated a key.
	Key *Key `json:"key,omitempty"`

	// Version: The version of the entity on the server after processing the
	// mutation. If
	// the mutation doesn't change anything on the server, then the version
	// will
	// be the version of the current entity or, if no entity is present, a
	// version
	// that is strictly greater than the version of any previous entity and
	// less
	// than the version of any possible future entity.
	Version int64 `json:"version,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "ConflictDetected") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConflictDetected") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MutationResult) MarshalJSON() ([]byte, error) {
	type noMethod MutationResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PartitionId: A partition ID identifies a grouping of entities. The
// grouping is always
// by project and namespace, however the namespace ID may be empty.
//
// A partition ID contains several dimensions:
// project ID and namespace ID.
//
// Partition dimensions:
//
// - May be "".
// - Must be valid UTF-8 bytes.
// - Must have values that match regex `[A-Za-z\d\.\-_]{1,100}`
// If the value of any dimension matches regex `__.*__`, the partition
// is
// reserved/read-only.
// A reserved/read-only partition ID is forbidden in certain
// documented
// contexts.
//
// Foreign partition IDs (in which the project ID does
// not match the context project ID ) are discouraged.
// Reads and writes of foreign partition IDs may fail if the project is
// not in an active state.
type PartitionId struct {
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

func (s *PartitionId) MarshalJSON() ([]byte, error) {
	type noMethod PartitionId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PathElement: A (kind, ID/name) pair used to construct a key path.
//
// If either name or ID is set, the element is complete.
// If neither is set, the element is incomplete.
type PathElement struct {
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

func (s *PathElement) MarshalJSON() ([]byte, error) {
	type noMethod PathElement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Projection: A representation of a property in a projection.
type Projection struct {
	// Property: The property to project.
	Property *PropertyReference `json:"property,omitempty"`

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

func (s *Projection) MarshalJSON() ([]byte, error) {
	type noMethod Projection
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PropertyFilter: A filter on a specific property.
type PropertyFilter struct {
	// Op: The operator to filter by.
	//
	// Possible values:
	//   "OPERATOR_UNSPECIFIED" - Unspecified. This value must not be used.
	//   "LESS_THAN" - Less than.
	//   "LESS_THAN_OR_EQUAL" - Less than or equal.
	//   "GREATER_THAN" - Greater than.
	//   "GREATER_THAN_OR_EQUAL" - Greater than or equal.
	//   "EQUAL" - Equal.
	//   "HAS_ANCESTOR" - Has ancestor.
	Op string `json:"op,omitempty"`

	// Property: The property to filter by.
	Property *PropertyReference `json:"property,omitempty"`

	// Value: The value to compare the property to.
	Value *Value `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Op") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Op") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PropertyFilter) MarshalJSON() ([]byte, error) {
	type noMethod PropertyFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PropertyOrder: The desired order for a specific property.
type PropertyOrder struct {
	// Direction: The direction to order by. Defaults to `ASCENDING`.
	//
	// Possible values:
	//   "DIRECTION_UNSPECIFIED" - Unspecified. This value must not be used.
	//   "ASCENDING" - Ascending.
	//   "DESCENDING" - Descending.
	Direction string `json:"direction,omitempty"`

	// Property: The property to order by.
	Property *PropertyReference `json:"property,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Direction") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Direction") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PropertyOrder) MarshalJSON() ([]byte, error) {
	type noMethod PropertyOrder
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PropertyReference: A reference to a property relative to the kind
// expressions.
type PropertyReference struct {
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

func (s *PropertyReference) MarshalJSON() ([]byte, error) {
	type noMethod PropertyReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Query: A query for entities.
type Query struct {
	// DistinctOn: The properties to make distinct. The query results will
	// contain the first
	// result for each distinct combination of values for the given
	// properties
	// (if empty, all results are returned).
	DistinctOn []*PropertyReference `json:"distinctOn,omitempty"`

	// EndCursor: An ending point for the query results. Query cursors
	// are
	// returned in query result batches and
	// [can only be used to limit the same
	// query](https://cloud.google.com/datastore/docs/concepts/queries#cursor
	// s_limits_and_offsets).
	EndCursor string `json:"endCursor,omitempty"`

	// Filter: The filter to apply.
	Filter *Filter `json:"filter,omitempty"`

	// Kind: The kinds to query (if empty, returns entities of all
	// kinds).
	// Currently at most 1 kind may be specified.
	Kind []*KindExpression `json:"kind,omitempty"`

	// Limit: The maximum number of results to return. Applies after all
	// other
	// constraints. Optional.
	// Unspecified is interpreted as no limit.
	// Must be >= 0 if specified.
	Limit int64 `json:"limit,omitempty"`

	// Offset: The number of results to skip. Applies before limit, but
	// after all other
	// constraints. Optional. Must be >= 0 if specified.
	Offset int64 `json:"offset,omitempty"`

	// Order: The order to apply to the query results (if empty, order is
	// unspecified).
	Order []*PropertyOrder `json:"order,omitempty"`

	// Projection: The projection to return. Defaults to returning all
	// properties.
	Projection []*Projection `json:"projection,omitempty"`

	// StartCursor: A starting point for the query results. Query cursors
	// are
	// returned in query result batches and
	// [can only be used to continue the same
	// query](https://cloud.google.com/datastore/docs/concepts/queries#cursor
	// s_limits_and_offsets).
	StartCursor string `json:"startCursor,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DistinctOn") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DistinctOn") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Query) MarshalJSON() ([]byte, error) {
	type noMethod Query
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QueryResultBatch: A batch of results produced by a query.
type QueryResultBatch struct {
	// EndCursor: A cursor that points to the position after the last result
	// in the batch.
	EndCursor string `json:"endCursor,omitempty"`

	// EntityResultType: The result type for every entity in
	// `entity_results`.
	//
	// Possible values:
	//   "RESULT_TYPE_UNSPECIFIED" - Unspecified. This value is never used.
	//   "FULL" - The key and properties.
	//   "PROJECTION" - A projected subset of properties. The entity may
	// have no key.
	//   "KEY_ONLY" - Only the key.
	EntityResultType string `json:"entityResultType,omitempty"`

	// EntityResults: The results for this batch.
	EntityResults []*EntityResult `json:"entityResults,omitempty"`

	// MoreResults: The state of the query after the current batch.
	//
	// Possible values:
	//   "MORE_RESULTS_TYPE_UNSPECIFIED" - Unspecified. This value is never
	// used.
	//   "NOT_FINISHED" - There may be additional batches to fetch from this
	// query.
	//   "MORE_RESULTS_AFTER_LIMIT" - The query is finished, but there may
	// be more results after the limit.
	//   "MORE_RESULTS_AFTER_CURSOR" - The query is finished, but there may
	// be more results after the end
	// cursor.
	//   "NO_MORE_RESULTS" - The query is finished, and there are no more
	// results.
	MoreResults string `json:"moreResults,omitempty"`

	// SkippedCursor: A cursor that points to the position after the last
	// skipped result.
	// Will be set when `skipped_results` != 0.
	SkippedCursor string `json:"skippedCursor,omitempty"`

	// SkippedResults: The number of results skipped, typically because of
	// an offset.
	SkippedResults int64 `json:"skippedResults,omitempty"`

	// SnapshotVersion: The version number of the snapshot this batch was
	// returned from.
	// This applies to the range of results from the query's `start_cursor`
	// (or
	// the beginning of the query if no cursor was given) to this
	// batch's
	// `end_cursor` (not the query's `end_cursor`).
	//
	// In a single transaction, subsequent query result batches for the same
	// query
	// can have a greater snapshot version number. Each batch's snapshot
	// version
	// is valid for all preceding batches.
	// The value will be zero for eventually consistent queries.
	SnapshotVersion int64 `json:"snapshotVersion,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "EndCursor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndCursor") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryResultBatch) MarshalJSON() ([]byte, error) {
	type noMethod QueryResultBatch
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReadOnly: Options specific to read-only transactions.
type ReadOnly struct {
}

// ReadOptions: The options shared by read requests.
type ReadOptions struct {
	// ReadConsistency: The non-transactional read consistency to
	// use.
	// Cannot be set to `STRONG` for global queries.
	//
	// Possible values:
	//   "READ_CONSISTENCY_UNSPECIFIED" - Unspecified. This value must not
	// be used.
	//   "STRONG" - Strong consistency.
	//   "EVENTUAL" - Eventual consistency.
	ReadConsistency string `json:"readConsistency,omitempty"`

	// Transaction: The identifier of the transaction in which to read.
	// A
	// transaction identifier is returned by a call
	// to
	// Datastore.BeginTransaction.
	Transaction string `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReadConsistency") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReadConsistency") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ReadOptions) MarshalJSON() ([]byte, error) {
	type noMethod ReadOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReadWrite: Options specific to read / write transactions.
type ReadWrite struct {
	// PreviousTransaction: The transaction identifier of the transaction
	// being retried.
	PreviousTransaction string `json:"previousTransaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PreviousTransaction")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PreviousTransaction") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ReadWrite) MarshalJSON() ([]byte, error) {
	type noMethod ReadWrite
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RollbackRequest: The request for Datastore.Rollback.
type RollbackRequest struct {
	// Transaction: The transaction identifier, returned by a call
	// to
	// Datastore.BeginTransaction.
	Transaction string `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Transaction") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Transaction") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RollbackRequest) MarshalJSON() ([]byte, error) {
	type noMethod RollbackRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RollbackResponse: The response for Datastore.Rollback.
// (an empty message).
type RollbackResponse struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// RunQueryRequest: The request for Datastore.RunQuery.
type RunQueryRequest struct {
	// GqlQuery: The GQL query to run.
	GqlQuery *GqlQuery `json:"gqlQuery,omitempty"`

	// PartitionId: Entities are partitioned into subsets, identified by a
	// partition ID.
	// Queries are scoped to a single partition.
	// This partition ID is normalized with the standard default
	// context
	// partition ID.
	PartitionId *PartitionId `json:"partitionId,omitempty"`

	// Query: The query to run.
	Query *Query `json:"query,omitempty"`

	// ReadOptions: The options for this query.
	ReadOptions *ReadOptions `json:"readOptions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "GqlQuery") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GqlQuery") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RunQueryRequest) MarshalJSON() ([]byte, error) {
	type noMethod RunQueryRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RunQueryResponse: The response for Datastore.RunQuery.
type RunQueryResponse struct {
	// Batch: A batch of query results (always present).
	Batch *QueryResultBatch `json:"batch,omitempty"`

	// Query: The parsed form of the `GqlQuery` from the request, if it was
	// set.
	Query *Query `json:"query,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Batch") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Batch") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RunQueryResponse) MarshalJSON() ([]byte, error) {
	type noMethod RunQueryResponse
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

// TransactionOptions: Options for beginning a new
// transaction.
//
// Transactions can be created explicitly with calls
// to
// Datastore.BeginTransaction or implicitly by
// setting
// ReadOptions.new_transaction in read requests.
type TransactionOptions struct {
	// ReadOnly: The transaction should only allow reads.
	ReadOnly *ReadOnly `json:"readOnly,omitempty"`

	// ReadWrite: The transaction should allow both reads and writes.
	ReadWrite *ReadWrite `json:"readWrite,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReadOnly") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReadOnly") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TransactionOptions) MarshalJSON() ([]byte, error) {
	type noMethod TransactionOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Value: A message that can hold any of the supported value types and
// associated
// metadata.
type Value struct {
	// ArrayValue: An array value.
	// Cannot contain another array value.
	// A `Value` instance that sets field `array_value` must not set
	// fields
	// `meaning` or `exclude_from_indexes`.
	ArrayValue *ArrayValue `json:"arrayValue,omitempty"`

	// BlobValue: A blob value.
	// May have at most 1,000,000 bytes.
	// When `exclude_from_indexes` is false, may have at most 1500 bytes.
	// In JSON requests, must be base64-encoded.
	BlobValue string `json:"blobValue,omitempty"`

	// BooleanValue: A boolean value.
	BooleanValue bool `json:"booleanValue,omitempty"`

	// DoubleValue: A double value.
	DoubleValue float64 `json:"doubleValue,omitempty"`

	// EntityValue: An entity value.
	//
	// - May have no key.
	// - May have a key with an incomplete key path.
	// - May have a reserved/read-only key.
	EntityValue *Entity `json:"entityValue,omitempty"`

	// ExcludeFromIndexes: If the value should be excluded from all indexes
	// including those defined
	// explicitly.
	ExcludeFromIndexes bool `json:"excludeFromIndexes,omitempty"`

	// GeoPointValue: A geo point value representing a point on the surface
	// of Earth.
	GeoPointValue *LatLng `json:"geoPointValue,omitempty"`

	// IntegerValue: An integer value.
	IntegerValue int64 `json:"integerValue,omitempty,string"`

	// KeyValue: A key value.
	KeyValue *Key `json:"keyValue,omitempty"`

	// Meaning: The `meaning` field should only be populated for backwards
	// compatibility.
	Meaning int64 `json:"meaning,omitempty"`

	// NullValue: A null value.
	//
	// Possible values:
	//   "NULL_VALUE" - Null value.
	NullValue string `json:"nullValue,omitempty"`

	// StringValue: A UTF-8 encoded string value.
	// When `exclude_from_indexes` is false (it is indexed) , may have at
	// most 1500 bytes.
	// Otherwise, may be set to at least 1,000,000 bytes.
	StringValue string `json:"stringValue,omitempty"`

	// TimestampValue: A timestamp value.
	// When stored in the Datastore, precise only to microseconds;
	// any additional precision is rounded down.
	TimestampValue string `json:"timestampValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArrayValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ArrayValue") to include in
	// API requests with the JSON null value. By default, fields with empty
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
		DoubleValue gensupport.JSONFloat64 `json:"doubleValue"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.DoubleValue = float64(s1.DoubleValue)
	return nil
}

// method id "datastore.projects.allocateIds":

type ProjectsAllocateIdsCall struct {
	s                  *Service
	projectId          string
	allocateidsrequest *AllocateIdsRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// AllocateIds: Allocates IDs for the given keys, which is useful for
// referencing an entity
// before it is inserted.
func (r *ProjectsService) AllocateIds(projectId string, allocateidsrequest *AllocateIdsRequest) *ProjectsAllocateIdsCall {
	c := &ProjectsAllocateIdsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.allocateidsrequest = allocateidsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsAllocateIdsCall) Fields(s ...googleapi.Field) *ProjectsAllocateIdsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsAllocateIdsCall) Context(ctx context.Context) *ProjectsAllocateIdsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsAllocateIdsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsAllocateIdsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.allocateidsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}:allocateIds")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.allocateIds" call.
// Exactly one of *AllocateIdsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AllocateIdsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsAllocateIdsCall) Do(opts ...googleapi.CallOption) (*AllocateIdsResponse, error) {
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
	ret := &AllocateIdsResponse{
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
	//   "description": "Allocates IDs for the given keys, which is useful for referencing an entity\nbefore it is inserted.",
	//   "flatPath": "v1/projects/{projectId}:allocateIds",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.allocateIds",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The ID of the project against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}:allocateIds",
	//   "request": {
	//     "$ref": "AllocateIdsRequest"
	//   },
	//   "response": {
	//     "$ref": "AllocateIdsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.beginTransaction":

type ProjectsBeginTransactionCall struct {
	s                       *Service
	projectId               string
	begintransactionrequest *BeginTransactionRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
	header_                 http.Header
}

// BeginTransaction: Begins a new transaction.
func (r *ProjectsService) BeginTransaction(projectId string, begintransactionrequest *BeginTransactionRequest) *ProjectsBeginTransactionCall {
	c := &ProjectsBeginTransactionCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.begintransactionrequest = begintransactionrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsBeginTransactionCall) Fields(s ...googleapi.Field) *ProjectsBeginTransactionCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsBeginTransactionCall) Context(ctx context.Context) *ProjectsBeginTransactionCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsBeginTransactionCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsBeginTransactionCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.begintransactionrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}:beginTransaction")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.beginTransaction" call.
// Exactly one of *BeginTransactionResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BeginTransactionResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsBeginTransactionCall) Do(opts ...googleapi.CallOption) (*BeginTransactionResponse, error) {
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
	ret := &BeginTransactionResponse{
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
	//   "description": "Begins a new transaction.",
	//   "flatPath": "v1/projects/{projectId}:beginTransaction",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.beginTransaction",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The ID of the project against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}:beginTransaction",
	//   "request": {
	//     "$ref": "BeginTransactionRequest"
	//   },
	//   "response": {
	//     "$ref": "BeginTransactionResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.commit":

type ProjectsCommitCall struct {
	s             *Service
	projectId     string
	commitrequest *CommitRequest
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Commit: Commits a transaction, optionally creating, deleting or
// modifying some
// entities.
func (r *ProjectsService) Commit(projectId string, commitrequest *CommitRequest) *ProjectsCommitCall {
	c := &ProjectsCommitCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.commitrequest = commitrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsCommitCall) Fields(s ...googleapi.Field) *ProjectsCommitCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsCommitCall) Context(ctx context.Context) *ProjectsCommitCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsCommitCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsCommitCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.commitrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}:commit")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.commit" call.
// Exactly one of *CommitResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CommitResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsCommitCall) Do(opts ...googleapi.CallOption) (*CommitResponse, error) {
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
	ret := &CommitResponse{
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
	//   "description": "Commits a transaction, optionally creating, deleting or modifying some\nentities.",
	//   "flatPath": "v1/projects/{projectId}:commit",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.commit",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The ID of the project against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}:commit",
	//   "request": {
	//     "$ref": "CommitRequest"
	//   },
	//   "response": {
	//     "$ref": "CommitResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.lookup":

type ProjectsLookupCall struct {
	s             *Service
	projectId     string
	lookuprequest *LookupRequest
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Lookup: Looks up entities by key.
func (r *ProjectsService) Lookup(projectId string, lookuprequest *LookupRequest) *ProjectsLookupCall {
	c := &ProjectsLookupCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.lookuprequest = lookuprequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLookupCall) Fields(s ...googleapi.Field) *ProjectsLookupCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLookupCall) Context(ctx context.Context) *ProjectsLookupCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLookupCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLookupCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.lookuprequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}:lookup")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.lookup" call.
// Exactly one of *LookupResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LookupResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLookupCall) Do(opts ...googleapi.CallOption) (*LookupResponse, error) {
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
	ret := &LookupResponse{
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
	//   "description": "Looks up entities by key.",
	//   "flatPath": "v1/projects/{projectId}:lookup",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.lookup",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The ID of the project against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}:lookup",
	//   "request": {
	//     "$ref": "LookupRequest"
	//   },
	//   "response": {
	//     "$ref": "LookupResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.rollback":

type ProjectsRollbackCall struct {
	s               *Service
	projectId       string
	rollbackrequest *RollbackRequest
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Rollback: Rolls back a transaction.
func (r *ProjectsService) Rollback(projectId string, rollbackrequest *RollbackRequest) *ProjectsRollbackCall {
	c := &ProjectsRollbackCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.rollbackrequest = rollbackrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRollbackCall) Fields(s ...googleapi.Field) *ProjectsRollbackCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRollbackCall) Context(ctx context.Context) *ProjectsRollbackCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRollbackCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRollbackCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.rollbackrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}:rollback")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.rollback" call.
// Exactly one of *RollbackResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RollbackResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsRollbackCall) Do(opts ...googleapi.CallOption) (*RollbackResponse, error) {
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
	ret := &RollbackResponse{
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
	//   "description": "Rolls back a transaction.",
	//   "flatPath": "v1/projects/{projectId}:rollback",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.rollback",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The ID of the project against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}:rollback",
	//   "request": {
	//     "$ref": "RollbackRequest"
	//   },
	//   "response": {
	//     "$ref": "RollbackResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.runQuery":

type ProjectsRunQueryCall struct {
	s               *Service
	projectId       string
	runqueryrequest *RunQueryRequest
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// RunQuery: Queries for entities.
func (r *ProjectsService) RunQuery(projectId string, runqueryrequest *RunQueryRequest) *ProjectsRunQueryCall {
	c := &ProjectsRunQueryCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.runqueryrequest = runqueryrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRunQueryCall) Fields(s ...googleapi.Field) *ProjectsRunQueryCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRunQueryCall) Context(ctx context.Context) *ProjectsRunQueryCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRunQueryCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRunQueryCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.runqueryrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}:runQuery")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.runQuery" call.
// Exactly one of *RunQueryResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RunQueryResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsRunQueryCall) Do(opts ...googleapi.CallOption) (*RunQueryResponse, error) {
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
	ret := &RunQueryResponse{
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
	//   "description": "Queries for entities.",
	//   "flatPath": "v1/projects/{projectId}:runQuery",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.runQuery",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The ID of the project against which to make the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}:runQuery",
	//   "request": {
	//     "$ref": "RunQueryRequest"
	//   },
	//   "response": {
	//     "$ref": "RunQueryResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.operations.cancel":

type ProjectsOperationsCancelCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Cancel: Starts asynchronous cancellation on a long-running operation.
//  The server
// makes a best effort to cancel the operation, but success is
// not
// guaranteed.  If the server doesn't support this method, it
// returns
// `google.rpc.Code.UNIMPLEMENTED`.  Clients can
// use
// Operations.GetOperation or
// other methods to check whether the cancellation succeeded or whether
// the
// operation completed despite cancellation. On successful
// cancellation,
// the operation is not deleted; instead, it becomes an operation
// with
// an Operation.error value with a google.rpc.Status.code of
// 1,
// corresponding to `Code.CANCELLED`.
func (r *ProjectsOperationsService) Cancel(name string) *ProjectsOperationsCancelCall {
	c := &ProjectsOperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsOperationsCancelCall) Fields(s ...googleapi.Field) *ProjectsOperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsOperationsCancelCall) Context(ctx context.Context) *ProjectsOperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsOperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsOperationsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.operations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsOperationsCancelCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Starts asynchronous cancellation on a long-running operation.  The server\nmakes a best effort to cancel the operation, but success is not\nguaranteed.  If the server doesn't support this method, it returns\n`google.rpc.Code.UNIMPLEMENTED`.  Clients can use\nOperations.GetOperation or\nother methods to check whether the cancellation succeeded or whether the\noperation completed despite cancellation. On successful cancellation,\nthe operation is not deleted; instead, it becomes an operation with\nan Operation.error value with a google.rpc.Status.code of 1,\ncorresponding to `Code.CANCELLED`.",
	//   "flatPath": "v1/projects/{projectsId}/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "datastore.projects.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:cancel",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.operations.delete":

type ProjectsOperationsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a long-running operation. This method indicates that
// the client is
// no longer interested in the operation result. It does not cancel
// the
// operation. If the server doesn't support this method, it
// returns
// `google.rpc.Code.UNIMPLEMENTED`.
func (r *ProjectsOperationsService) Delete(name string) *ProjectsOperationsDeleteCall {
	c := &ProjectsOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsOperationsDeleteCall) Fields(s ...googleapi.Field) *ProjectsOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsOperationsDeleteCall) Context(ctx context.Context) *ProjectsOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.operations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsOperationsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Deletes a long-running operation. This method indicates that the client is\nno longer interested in the operation result. It does not cancel the\noperation. If the server doesn't support this method, it returns\n`google.rpc.Code.UNIMPLEMENTED`.",
	//   "flatPath": "v1/projects/{projectsId}/operations/{operationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "datastore.projects.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.operations.get":

type ProjectsOperationsGetCall struct {
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
func (r *ProjectsOperationsService) Get(name string) *ProjectsOperationsGetCall {
	c := &ProjectsOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsOperationsGetCall) Fields(s ...googleapi.Field) *ProjectsOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsOperationsGetCall) IfNoneMatch(entityTag string) *ProjectsOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsOperationsGetCall) Context(ctx context.Context) *ProjectsOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.operations.get" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsOperationsGetCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	//   "flatPath": "v1/projects/{projectsId}/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "datastore.projects.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "GoogleLongrunningOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// method id "datastore.projects.operations.list":

type ProjectsOperationsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists operations that match the specified filter in the
// request. If the
// server doesn't support this method, it returns
// `UNIMPLEMENTED`.
//
// NOTE: the `name` binding allows API services to override the
// binding
// to use different resource name schemes, such as `users/*/operations`.
// To
// override the binding, API services can add a binding such
// as
// "/v1/{name=users/*}/operations" to their service configuration.
// For backwards compatibility, the default name includes the
// operations
// collection id, however overriding users must ensure the name
// binding
// is the parent resource, without the operations collection id.
func (r *ProjectsOperationsService) List(name string) *ProjectsOperationsListCall {
	c := &ProjectsOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": The standard list
// filter.
func (c *ProjectsOperationsListCall) Filter(filter string) *ProjectsOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The standard list
// page size.
func (c *ProjectsOperationsListCall) PageSize(pageSize int64) *ProjectsOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *ProjectsOperationsListCall) PageToken(pageToken string) *ProjectsOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsOperationsListCall) Fields(s ...googleapi.Field) *ProjectsOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsOperationsListCall) IfNoneMatch(entityTag string) *ProjectsOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsOperationsListCall) Context(ctx context.Context) *ProjectsOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsOperationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "datastore.projects.operations.list" call.
// Exactly one of *GoogleLongrunningListOperationsResponse or error will
// be non-nil. Any non-2xx status code is an error. Response headers are
// in either
// *GoogleLongrunningListOperationsResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsOperationsListCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningListOperationsResponse, error) {
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
	//   "description": "Lists operations that match the specified filter in the request. If the\nserver doesn't support this method, it returns `UNIMPLEMENTED`.\n\nNOTE: the `name` binding allows API services to override the binding\nto use different resource name schemes, such as `users/*/operations`. To\noverride the binding, API services can add a binding such as\n`\"/v1/{name=users/*}/operations\"` to their service configuration.\nFor backwards compatibility, the default name includes the operations\ncollection id, however overriding users must ensure the name binding\nis the parent resource, without the operations collection id.",
	//   "flatPath": "v1/projects/{projectsId}/operations",
	//   "httpMethod": "GET",
	//   "id": "datastore.projects.operations.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "The standard list filter.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The name of the operation's parent resource.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The standard list page size.",
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
	//   "path": "v1/{+name}/operations",
	//   "response": {
	//     "$ref": "GoogleLongrunningListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsOperationsListCall) Pages(ctx context.Context, f func(*GoogleLongrunningListOperationsResponse) error) error {
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
