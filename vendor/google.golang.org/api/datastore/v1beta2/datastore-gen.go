// Package datastore provides access to the Google Cloud Datastore API.
//
// See https://developers.google.com/datastore/
//
// Usage example:
//
//   import "google.golang.org/api/datastore/v1beta2"
//   ...
//   datastoreService, err := datastore.New(oauthHttpClient)
package datastore // import "google.golang.org/api/datastore/v1beta2"

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

const apiId = "datastore:v1beta2"
const apiName = "datastore"
const apiVersion = "v1beta2"
const basePath = "https://www.googleapis.com/datastore/v1beta2/datasets/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View and manage your Google Cloud Datastore data
	DatastoreScope = "https://www.googleapis.com/auth/datastore"

	// View your email address
	UserinfoEmailScope = "https://www.googleapis.com/auth/userinfo.email"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Datasets = NewDatasetsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Datasets *DatasetsService
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

type AllocateIdsRequest struct {
	// Keys: A list of keys with incomplete key paths to allocate IDs for.
	// No key may be reserved/read-only.
	Keys []*Key `json:"keys,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Keys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AllocateIdsRequest) MarshalJSON() ([]byte, error) {
	type noMethod AllocateIdsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AllocateIdsResponse struct {
	Header *ResponseHeader `json:"header,omitempty"`

	// Keys: The keys specified in the request (in the same order), each
	// with its key path completed with a newly allocated ID.
	Keys []*Key `json:"keys,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Header") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AllocateIdsResponse) MarshalJSON() ([]byte, error) {
	type noMethod AllocateIdsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type BeginTransactionRequest struct {
	// IsolationLevel: The transaction isolation level. Either snapshot or
	// serializable. The default isolation level is snapshot isolation,
	// which means that another transaction may not concurrently modify the
	// data that is modified by this transaction. Optionally, a transaction
	// can request to be made serializable which means that another
	// transaction cannot concurrently modify the data that is read or
	// modified by this transaction.
	//
	// Possible values:
	//   "SERIALIZABLE"
	//   "SNAPSHOT"
	IsolationLevel string `json:"isolationLevel,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IsolationLevel") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BeginTransactionRequest) MarshalJSON() ([]byte, error) {
	type noMethod BeginTransactionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type BeginTransactionResponse struct {
	Header *ResponseHeader `json:"header,omitempty"`

	// Transaction: The transaction identifier (always present).
	Transaction string `json:"transaction,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Header") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BeginTransactionResponse) MarshalJSON() ([]byte, error) {
	type noMethod BeginTransactionResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CommitRequest struct {
	IgnoreReadOnly bool `json:"ignoreReadOnly,omitempty"`

	// Mode: The type of commit to perform. Either TRANSACTIONAL or
	// NON_TRANSACTIONAL.
	//
	// Possible values:
	//   "NON_TRANSACTIONAL"
	//   "TRANSACTIONAL"
	Mode string `json:"mode,omitempty"`

	// Mutation: The mutation to perform. Optional.
	Mutation *Mutation `json:"mutation,omitempty"`

	// Transaction: The transaction identifier, returned by a call to
	// beginTransaction. Must be set when mode is TRANSACTIONAL.
	Transaction string `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IgnoreReadOnly") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CommitRequest) MarshalJSON() ([]byte, error) {
	type noMethod CommitRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type CommitResponse struct {
	Header *ResponseHeader `json:"header,omitempty"`

	// MutationResult: The result of performing the mutation (if any).
	MutationResult *MutationResult `json:"mutationResult,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Header") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CommitResponse) MarshalJSON() ([]byte, error) {
	type noMethod CommitResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CompositeFilter: A filter that merges the multiple other filters
// using the given operation.
type CompositeFilter struct {
	// Filters: The list of filters to combine. Must contain at least one
	// filter.
	Filters []*Filter `json:"filters,omitempty"`

	// Operator: The operator for combining multiple filters. Only "and" is
	// currently supported.
	//
	// Possible values:
	//   "AND"
	Operator string `json:"operator,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CompositeFilter) MarshalJSON() ([]byte, error) {
	type noMethod CompositeFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Entity: An entity.
type Entity struct {
	// Key: The entity's key.
	//
	// An entity must have a key, unless otherwise documented (for example,
	// an entity in Value.entityValue may have no key). An entity's kind is
	// its key's path's last element's kind, or null if it has no key.
	Key *Key `json:"key,omitempty"`

	// Properties: The entity's properties.
	Properties map[string]Property `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Entity) MarshalJSON() ([]byte, error) {
	type noMethod Entity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// EntityResult: The result of fetching an entity from the datastore.
type EntityResult struct {
	// Entity: The resulting entity.
	Entity *Entity `json:"entity,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entity") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EntityResult) MarshalJSON() ([]byte, error) {
	type noMethod EntityResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Filter: A holder for any type of filter. Exactly one field should be
// specified.
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
}

func (s *Filter) MarshalJSON() ([]byte, error) {
	type noMethod Filter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GqlQuery: A GQL query.
type GqlQuery struct {
	// AllowLiteral: When false, the query string must not contain a
	// literal.
	AllowLiteral bool `json:"allowLiteral,omitempty"`

	// NameArgs: A named argument must set field GqlQueryArg.name. No two
	// named arguments may have the same name. For each non-reserved named
	// binding site in the query string, there must be a named argument with
	// that name, but not necessarily the inverse.
	NameArgs []*GqlQueryArg `json:"nameArgs,omitempty"`

	// NumberArgs: Numbered binding site @1 references the first numbered
	// argument, effectively using 1-based indexing, rather than the usual
	// 0. A numbered argument must NOT set field GqlQueryArg.name. For each
	// binding site numbered i in query_string, there must be an ith
	// numbered argument. The inverse must also be true.
	NumberArgs []*GqlQueryArg `json:"numberArgs,omitempty"`

	// QueryString: The query string.
	QueryString string `json:"queryString,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AllowLiteral") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GqlQuery) MarshalJSON() ([]byte, error) {
	type noMethod GqlQuery
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GqlQueryArg: A binding argument for a GQL query.
type GqlQueryArg struct {
	Cursor string `json:"cursor,omitempty"`

	// Name: Must match regex "[A-Za-z_$][A-Za-z_$0-9]*". Must not match
	// regex "__.*__". Must not be "".
	Name string `json:"name,omitempty"`

	Value *Value `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cursor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GqlQueryArg) MarshalJSON() ([]byte, error) {
	type noMethod GqlQueryArg
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Key: A unique identifier for an entity.
type Key struct {
	// PartitionId: Entities are partitioned into subsets, currently
	// identified by a dataset (usually implicitly specified by the project)
	// and namespace ID. Queries are scoped to a single partition.
	PartitionId *PartitionId `json:"partitionId,omitempty"`

	// Path: The entity path. An entity path consists of one or more
	// elements composed of a kind and a string or numerical identifier,
	// which identify entities. The first element identifies a root entity,
	// the second element identifies a child of the root entity, the third
	// element a child of the second entity, and so forth. The entities
	// identified by all prefixes of the path are called the element's
	// ancestors. An entity path is always fully complete: ALL of the
	// entity's ancestors are required to be in the path along with the
	// entity identifier itself. The only exception is that in some
	// documented cases, the identifier in the last path element (for the
	// entity) itself may be omitted. A path can never be empty. The path
	// can have at most 100 elements.
	Path []*KeyPathElement `json:"path,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PartitionId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Key) MarshalJSON() ([]byte, error) {
	type noMethod Key
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// KeyPathElement: A (kind, ID/name) pair used to construct a key
// path.
//
// At most one of name or ID may be set. If either is set, the element
// is complete. If neither is set, the element is incomplete.
type KeyPathElement struct {
	// Id: The ID of the entity. Never equal to zero. Values less than zero
	// are discouraged and will not be supported in the future.
	Id int64 `json:"id,omitempty,string"`

	// Kind: The kind of the entity. A kind matching regex "__.*__" is
	// reserved/read-only. A kind must not contain more than 500 characters.
	// Cannot be "".
	Kind string `json:"kind,omitempty"`

	// Name: The name of the entity. A name matching regex "__.*__" is
	// reserved/read-only. A name must not be more than 500 characters.
	// Cannot be "".
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *KeyPathElement) MarshalJSON() ([]byte, error) {
	type noMethod KeyPathElement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *KindExpression) MarshalJSON() ([]byte, error) {
	type noMethod KindExpression
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type LookupRequest struct {
	// Keys: Keys of entities to look up from the datastore.
	Keys []*Key `json:"keys,omitempty"`

	// ReadOptions: Options for this lookup request. Optional.
	ReadOptions *ReadOptions `json:"readOptions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Keys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LookupRequest) MarshalJSON() ([]byte, error) {
	type noMethod LookupRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type LookupResponse struct {
	// Deferred: A list of keys that were not looked up due to resource
	// constraints.
	Deferred []*Key `json:"deferred,omitempty"`

	// Found: Entities found.
	Found []*EntityResult `json:"found,omitempty"`

	Header *ResponseHeader `json:"header,omitempty"`

	// Missing: Entities not found, with only the key populated.
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
}

func (s *LookupResponse) MarshalJSON() ([]byte, error) {
	type noMethod LookupResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Mutation: A set of changes to apply.
type Mutation struct {
	// Delete: Keys of entities to delete. Each key must have a complete key
	// path and must not be reserved/read-only.
	Delete []*Key `json:"delete,omitempty"`

	// Force: Ignore a user specified read-only period. Optional.
	Force bool `json:"force,omitempty"`

	// Insert: Entities to insert. Each inserted entity's key must have a
	// complete path and must not be reserved/read-only.
	Insert []*Entity `json:"insert,omitempty"`

	// InsertAutoId: Insert entities with a newly allocated ID. Each
	// inserted entity's key must omit the final identifier in its path and
	// must not be reserved/read-only.
	InsertAutoId []*Entity `json:"insertAutoId,omitempty"`

	// Update: Entities to update. Each updated entity's key must have a
	// complete path and must not be reserved/read-only.
	Update []*Entity `json:"update,omitempty"`

	// Upsert: Entities to upsert. Each upserted entity's key must have a
	// complete path and must not be reserved/read-only.
	Upsert []*Entity `json:"upsert,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Delete") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Mutation) MarshalJSON() ([]byte, error) {
	type noMethod Mutation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type MutationResult struct {
	// IndexUpdates: Number of index writes.
	IndexUpdates int64 `json:"indexUpdates,omitempty"`

	// InsertAutoIdKeys: Keys for insertAutoId entities. One per entity from
	// the request, in the same order.
	InsertAutoIdKeys []*Key `json:"insertAutoIdKeys,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IndexUpdates") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MutationResult) MarshalJSON() ([]byte, error) {
	type noMethod MutationResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PartitionId: An identifier for a particular subset of
// entities.
//
// Entities are partitioned into various subsets, each used by different
// datasets and different namespaces within a dataset and so forth.
type PartitionId struct {
	// DatasetId: The dataset ID.
	DatasetId string `json:"datasetId,omitempty"`

	// Namespace: The namespace.
	Namespace string `json:"namespace,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PartitionId) MarshalJSON() ([]byte, error) {
	type noMethod PartitionId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Property: An entity property.
type Property struct {
	// BlobKeyValue: A blob key value.
	BlobKeyValue *string `json:"blobKeyValue,omitempty"`

	// BlobValue: A blob value. May be a maximum of 1,000,000 bytes. When
	// indexed is true, may have at most 500 bytes.
	BlobValue *string `json:"blobValue,omitempty"`

	// BooleanValue: A boolean value.
	BooleanValue *bool `json:"booleanValue,omitempty"`

	// DateTimeValue: A timestamp value.
	DateTimeValue *string `json:"dateTimeValue,omitempty"`

	// DoubleValue: A double value.
	DoubleValue *float64 `json:"doubleValue,omitempty"`

	// EntityValue: An entity value. May have no key. May have a key with an
	// incomplete key path. May have a reserved/read-only key.
	EntityValue *Entity `json:"entityValue,omitempty"`

	// Indexed: If the value should be indexed.
	//
	// The indexed property may be set for a null value. When indexed is
	// true, stringValue is limited to 500 characters and the blob value is
	// limited to 500 bytes. Input values by default have indexed set to
	// true; however, you can explicitly set indexed to true if you want.
	// (An output value never has indexed explicitly set to true.) If a
	// value is itself an entity, it cannot have indexed set to true.
	Indexed *bool `json:"indexed,omitempty"`

	// IntegerValue: An integer value.
	IntegerValue *int64 `json:"integerValue,omitempty,string"`

	// KeyValue: A key value.
	KeyValue *Key `json:"keyValue,omitempty"`

	// ListValue: A list value. Cannot contain another list value. A Value
	// instance that sets field list_value must not set field meaning or
	// field indexed.
	ListValue []*Value `json:"listValue,omitempty"`

	// Meaning: The meaning field is reserved and should not be used.
	Meaning int64 `json:"meaning,omitempty"`

	// StringValue: A UTF-8 encoded string value. When indexed is true, may
	// have at most 500 characters.
	StringValue *string `json:"stringValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BlobKeyValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Property) MarshalJSON() ([]byte, error) {
	type noMethod Property
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PropertyExpression: A representation of a property in a projection.
type PropertyExpression struct {
	// AggregationFunction: The aggregation function to apply to the
	// property. Optional. Can only be used when grouping by at least one
	// property. Must then be set on all properties in the projection that
	// are not being grouped by. Aggregation functions: first selects the
	// first result as determined by the query's order.
	//
	// Possible values:
	//   "FIRST"
	AggregationFunction string `json:"aggregationFunction,omitempty"`

	// Property: The property to project.
	Property *PropertyReference `json:"property,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AggregationFunction")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PropertyExpression) MarshalJSON() ([]byte, error) {
	type noMethod PropertyExpression
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PropertyFilter: A filter on a specific property.
type PropertyFilter struct {
	// Operator: The operator to filter by. One of lessThan,
	// lessThanOrEqual, greaterThan, greaterThanOrEqual, equal, or
	// hasAncestor.
	//
	// Possible values:
	//   "EQUAL"
	//   "GREATER_THAN"
	//   "GREATER_THAN_OR_EQUAL"
	//   "HAS_ANCESTOR"
	//   "LESS_THAN"
	//   "LESS_THAN_OR_EQUAL"
	Operator string `json:"operator,omitempty"`

	// Property: The property to filter by.
	Property *PropertyReference `json:"property,omitempty"`

	// Value: The value to compare the property to.
	Value *Value `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Operator") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PropertyFilter) MarshalJSON() ([]byte, error) {
	type noMethod PropertyFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PropertyOrder: The desired order for a specific property.
type PropertyOrder struct {
	// Direction: The direction to order by. One of ascending or descending.
	// Optional, defaults to ascending.
	//
	// Possible values:
	//   "ASCENDING"
	//   "DESCENDING"
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
}

func (s *PropertyOrder) MarshalJSON() ([]byte, error) {
	type noMethod PropertyOrder
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PropertyReference: A reference to a property relative to the kind
// expressions.
type PropertyReference struct {
	// Name: The name of the property.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PropertyReference) MarshalJSON() ([]byte, error) {
	type noMethod PropertyReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Query: A query.
type Query struct {
	// EndCursor: An ending point for the query results. Optional. Query
	// cursors are returned in query result batches.
	EndCursor string `json:"endCursor,omitempty"`

	// Filter: The filter to apply (optional).
	Filter *Filter `json:"filter,omitempty"`

	// GroupBy: The properties to group by (if empty, no grouping is applied
	// to the result set).
	GroupBy []*PropertyReference `json:"groupBy,omitempty"`

	// Kinds: The kinds to query (if empty, returns entities from all
	// kinds).
	Kinds []*KindExpression `json:"kinds,omitempty"`

	// Limit: The maximum number of results to return. Applies after all
	// other constraints. Optional.
	Limit int64 `json:"limit,omitempty"`

	// Offset: The number of results to skip. Applies before limit, but
	// after all other constraints (optional, defaults to 0).
	Offset int64 `json:"offset,omitempty"`

	// Order: The order to apply to the query results (if empty, order is
	// unspecified).
	Order []*PropertyOrder `json:"order,omitempty"`

	// Projection: The projection to return. If not set the entire entity is
	// returned.
	Projection []*PropertyExpression `json:"projection,omitempty"`

	// StartCursor: A starting point for the query results. Optional. Query
	// cursors are returned in query result batches.
	StartCursor string `json:"startCursor,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndCursor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Query) MarshalJSON() ([]byte, error) {
	type noMethod Query
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// QueryResultBatch: A batch of results produced by a query.
type QueryResultBatch struct {
	// EndCursor: A cursor that points to the position after the last result
	// in the batch. May be absent. TODO(arfuller): Once all plans produce
	// cursors update documentation here.
	EndCursor string `json:"endCursor,omitempty"`

	// EntityResultType: The result type for every entity in entityResults.
	// full for full entities, projection for entities with only projected
	// properties, keyOnly for entities with only a key.
	//
	// Possible values:
	//   "FULL"
	//   "KEY_ONLY"
	//   "PROJECTION"
	EntityResultType string `json:"entityResultType,omitempty"`

	// EntityResults: The results for this batch.
	EntityResults []*EntityResult `json:"entityResults,omitempty"`

	// MoreResults: The state of the query after the current batch. One of
	// notFinished, moreResultsAfterLimit, noMoreResults.
	//
	// Possible values:
	//   "MORE_RESULTS_AFTER_LIMIT"
	//   "NOT_FINISHED"
	//   "NO_MORE_RESULTS"
	MoreResults string `json:"moreResults,omitempty"`

	// SkippedResults: The number of results skipped because of
	// Query.offset.
	SkippedResults int64 `json:"skippedResults,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndCursor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *QueryResultBatch) MarshalJSON() ([]byte, error) {
	type noMethod QueryResultBatch
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ReadOptions struct {
	// ReadConsistency: The read consistency to use. One of default, strong,
	// or eventual. Cannot be set when transaction is set. Lookup and
	// ancestor queries default to strong, global queries default to
	// eventual and cannot be set to strong. Optional. Default is default.
	//
	// Possible values:
	//   "DEFAULT"
	//   "EVENTUAL"
	//   "STRONG"
	ReadConsistency string `json:"readConsistency,omitempty"`

	// Transaction: The transaction to use. Optional.
	Transaction string `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReadConsistency") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReadOptions) MarshalJSON() ([]byte, error) {
	type noMethod ReadOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ResponseHeader struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "datastore#responseHeader".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ResponseHeader) MarshalJSON() ([]byte, error) {
	type noMethod ResponseHeader
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type RollbackRequest struct {
	// Transaction: The transaction identifier, returned by a call to
	// beginTransaction.
	Transaction string `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Transaction") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *RollbackRequest) MarshalJSON() ([]byte, error) {
	type noMethod RollbackRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type RollbackResponse struct {
	Header *ResponseHeader `json:"header,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Header") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *RollbackResponse) MarshalJSON() ([]byte, error) {
	type noMethod RollbackResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type RunQueryRequest struct {
	// GqlQuery: The GQL query to run. Either this field or field query must
	// be set, but not both.
	GqlQuery *GqlQuery `json:"gqlQuery,omitempty"`

	// PartitionId: Entities are partitioned into subsets, identified by a
	// dataset (usually implicitly specified by the project) and namespace
	// ID. Queries are scoped to a single partition. This partition ID is
	// normalized with the standard default context partition ID, but all
	// other partition IDs in RunQueryRequest are normalized with this
	// partition ID as the context partition ID.
	PartitionId *PartitionId `json:"partitionId,omitempty"`

	// Query: The query to run. Either this field or field gql_query must be
	// set, but not both.
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
}

func (s *RunQueryRequest) MarshalJSON() ([]byte, error) {
	type noMethod RunQueryRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type RunQueryResponse struct {
	// Batch: A batch of query results (always present).
	Batch *QueryResultBatch `json:"batch,omitempty"`

	Header *ResponseHeader `json:"header,omitempty"`

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
}

func (s *RunQueryResponse) MarshalJSON() ([]byte, error) {
	type noMethod RunQueryResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Value: A message that can hold any of the supported value types and
// associated metadata.
type Value struct {
	// BlobKeyValue: A blob key value.
	BlobKeyValue string `json:"blobKeyValue,omitempty"`

	// BlobValue: A blob value. May be a maximum of 1,000,000 bytes. When
	// indexed is true, may have at most 500 bytes.
	BlobValue string `json:"blobValue,omitempty"`

	// BooleanValue: A boolean value.
	BooleanValue bool `json:"booleanValue,omitempty"`

	// DateTimeValue: A timestamp value.
	DateTimeValue string `json:"dateTimeValue,omitempty"`

	// DoubleValue: A double value.
	DoubleValue float64 `json:"doubleValue,omitempty"`

	// EntityValue: An entity value. May have no key. May have a key with an
	// incomplete key path. May have a reserved/read-only key.
	EntityValue *Entity `json:"entityValue,omitempty"`

	// Indexed: If the value should be indexed.
	//
	// The indexed property may be set for a null value. When indexed is
	// true, stringValue is limited to 500 characters and the blob value is
	// limited to 500 bytes. Input values by default have indexed set to
	// true; however, you can explicitly set indexed to true if you want.
	// (An output value never has indexed explicitly set to true.) If a
	// value is itself an entity, it cannot have indexed set to true.
	Indexed bool `json:"indexed,omitempty"`

	// IntegerValue: An integer value.
	IntegerValue int64 `json:"integerValue,omitempty,string"`

	// KeyValue: A key value.
	KeyValue *Key `json:"keyValue,omitempty"`

	// ListValue: A list value. Cannot contain another list value. A Value
	// instance that sets field list_value must not set field meaning or
	// field indexed.
	ListValue []*Value `json:"listValue,omitempty"`

	// Meaning: The meaning field is reserved and should not be used.
	Meaning int64 `json:"meaning,omitempty"`

	// StringValue: A UTF-8 encoded string value. When indexed is true, may
	// have at most 500 characters.
	StringValue string `json:"stringValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BlobKeyValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Value) MarshalJSON() ([]byte, error) {
	type noMethod Value
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "datastore.datasets.allocateIds":

type DatasetsAllocateIdsCall struct {
	s                  *Service
	datasetId          string
	allocateidsrequest *AllocateIdsRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
}

// AllocateIds: Allocate IDs for incomplete keys (useful for referencing
// an entity before it is inserted).
func (r *DatasetsService) AllocateIds(datasetId string, allocateidsrequest *AllocateIdsRequest) *DatasetsAllocateIdsCall {
	c := &DatasetsAllocateIdsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.allocateidsrequest = allocateidsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatasetsAllocateIdsCall) QuotaUser(quotaUser string) *DatasetsAllocateIdsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatasetsAllocateIdsCall) UserIP(userIP string) *DatasetsAllocateIdsCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsAllocateIdsCall) Fields(s ...googleapi.Field) *DatasetsAllocateIdsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsAllocateIdsCall) Context(ctx context.Context) *DatasetsAllocateIdsCall {
	c.ctx_ = ctx
	return c
}

func (c *DatasetsAllocateIdsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.allocateidsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{datasetId}/allocateIds")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "datastore.datasets.allocateIds" call.
// Exactly one of *AllocateIdsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AllocateIdsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsAllocateIdsCall) Do() (*AllocateIdsResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Allocate IDs for incomplete keys (useful for referencing an entity before it is inserted).",
	//   "httpMethod": "POST",
	//   "id": "datastore.datasets.allocateIds",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Identifies the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{datasetId}/allocateIds",
	//   "request": {
	//     "$ref": "AllocateIdsRequest"
	//   },
	//   "response": {
	//     "$ref": "AllocateIdsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore",
	//     "https://www.googleapis.com/auth/userinfo.email"
	//   ]
	// }

}

// method id "datastore.datasets.beginTransaction":

type DatasetsBeginTransactionCall struct {
	s                       *Service
	datasetId               string
	begintransactionrequest *BeginTransactionRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
}

// BeginTransaction: Begin a new transaction.
func (r *DatasetsService) BeginTransaction(datasetId string, begintransactionrequest *BeginTransactionRequest) *DatasetsBeginTransactionCall {
	c := &DatasetsBeginTransactionCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.begintransactionrequest = begintransactionrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatasetsBeginTransactionCall) QuotaUser(quotaUser string) *DatasetsBeginTransactionCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatasetsBeginTransactionCall) UserIP(userIP string) *DatasetsBeginTransactionCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsBeginTransactionCall) Fields(s ...googleapi.Field) *DatasetsBeginTransactionCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsBeginTransactionCall) Context(ctx context.Context) *DatasetsBeginTransactionCall {
	c.ctx_ = ctx
	return c
}

func (c *DatasetsBeginTransactionCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.begintransactionrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{datasetId}/beginTransaction")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "datastore.datasets.beginTransaction" call.
// Exactly one of *BeginTransactionResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BeginTransactionResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsBeginTransactionCall) Do() (*BeginTransactionResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Begin a new transaction.",
	//   "httpMethod": "POST",
	//   "id": "datastore.datasets.beginTransaction",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Identifies the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{datasetId}/beginTransaction",
	//   "request": {
	//     "$ref": "BeginTransactionRequest"
	//   },
	//   "response": {
	//     "$ref": "BeginTransactionResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore",
	//     "https://www.googleapis.com/auth/userinfo.email"
	//   ]
	// }

}

// method id "datastore.datasets.commit":

type DatasetsCommitCall struct {
	s             *Service
	datasetId     string
	commitrequest *CommitRequest
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Commit: Commit a transaction, optionally creating, deleting or
// modifying some entities.
func (r *DatasetsService) Commit(datasetId string, commitrequest *CommitRequest) *DatasetsCommitCall {
	c := &DatasetsCommitCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.commitrequest = commitrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatasetsCommitCall) QuotaUser(quotaUser string) *DatasetsCommitCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatasetsCommitCall) UserIP(userIP string) *DatasetsCommitCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsCommitCall) Fields(s ...googleapi.Field) *DatasetsCommitCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsCommitCall) Context(ctx context.Context) *DatasetsCommitCall {
	c.ctx_ = ctx
	return c
}

func (c *DatasetsCommitCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.commitrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{datasetId}/commit")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "datastore.datasets.commit" call.
// Exactly one of *CommitResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CommitResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsCommitCall) Do() (*CommitResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Commit a transaction, optionally creating, deleting or modifying some entities.",
	//   "httpMethod": "POST",
	//   "id": "datastore.datasets.commit",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Identifies the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{datasetId}/commit",
	//   "request": {
	//     "$ref": "CommitRequest"
	//   },
	//   "response": {
	//     "$ref": "CommitResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore",
	//     "https://www.googleapis.com/auth/userinfo.email"
	//   ]
	// }

}

// method id "datastore.datasets.lookup":

type DatasetsLookupCall struct {
	s             *Service
	datasetId     string
	lookuprequest *LookupRequest
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Lookup: Look up some entities by key.
func (r *DatasetsService) Lookup(datasetId string, lookuprequest *LookupRequest) *DatasetsLookupCall {
	c := &DatasetsLookupCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.lookuprequest = lookuprequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatasetsLookupCall) QuotaUser(quotaUser string) *DatasetsLookupCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatasetsLookupCall) UserIP(userIP string) *DatasetsLookupCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsLookupCall) Fields(s ...googleapi.Field) *DatasetsLookupCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsLookupCall) Context(ctx context.Context) *DatasetsLookupCall {
	c.ctx_ = ctx
	return c
}

func (c *DatasetsLookupCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.lookuprequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{datasetId}/lookup")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "datastore.datasets.lookup" call.
// Exactly one of *LookupResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LookupResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsLookupCall) Do() (*LookupResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Look up some entities by key.",
	//   "httpMethod": "POST",
	//   "id": "datastore.datasets.lookup",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Identifies the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{datasetId}/lookup",
	//   "request": {
	//     "$ref": "LookupRequest"
	//   },
	//   "response": {
	//     "$ref": "LookupResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore",
	//     "https://www.googleapis.com/auth/userinfo.email"
	//   ]
	// }

}

// method id "datastore.datasets.rollback":

type DatasetsRollbackCall struct {
	s               *Service
	datasetId       string
	rollbackrequest *RollbackRequest
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// Rollback: Roll back a transaction.
func (r *DatasetsService) Rollback(datasetId string, rollbackrequest *RollbackRequest) *DatasetsRollbackCall {
	c := &DatasetsRollbackCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.rollbackrequest = rollbackrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatasetsRollbackCall) QuotaUser(quotaUser string) *DatasetsRollbackCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatasetsRollbackCall) UserIP(userIP string) *DatasetsRollbackCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsRollbackCall) Fields(s ...googleapi.Field) *DatasetsRollbackCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsRollbackCall) Context(ctx context.Context) *DatasetsRollbackCall {
	c.ctx_ = ctx
	return c
}

func (c *DatasetsRollbackCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.rollbackrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{datasetId}/rollback")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "datastore.datasets.rollback" call.
// Exactly one of *RollbackResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RollbackResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsRollbackCall) Do() (*RollbackResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Roll back a transaction.",
	//   "httpMethod": "POST",
	//   "id": "datastore.datasets.rollback",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Identifies the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{datasetId}/rollback",
	//   "request": {
	//     "$ref": "RollbackRequest"
	//   },
	//   "response": {
	//     "$ref": "RollbackResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore",
	//     "https://www.googleapis.com/auth/userinfo.email"
	//   ]
	// }

}

// method id "datastore.datasets.runQuery":

type DatasetsRunQueryCall struct {
	s               *Service
	datasetId       string
	runqueryrequest *RunQueryRequest
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// RunQuery: Query for entities.
func (r *DatasetsService) RunQuery(datasetId string, runqueryrequest *RunQueryRequest) *DatasetsRunQueryCall {
	c := &DatasetsRunQueryCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.runqueryrequest = runqueryrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DatasetsRunQueryCall) QuotaUser(quotaUser string) *DatasetsRunQueryCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DatasetsRunQueryCall) UserIP(userIP string) *DatasetsRunQueryCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsRunQueryCall) Fields(s ...googleapi.Field) *DatasetsRunQueryCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsRunQueryCall) Context(ctx context.Context) *DatasetsRunQueryCall {
	c.ctx_ = ctx
	return c
}

func (c *DatasetsRunQueryCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.runqueryrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{datasetId}/runQuery")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "datastore.datasets.runQuery" call.
// Exactly one of *RunQueryResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RunQueryResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsRunQueryCall) Do() (*RunQueryResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Query for entities.",
	//   "httpMethod": "POST",
	//   "id": "datastore.datasets.runQuery",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Identifies the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{datasetId}/runQuery",
	//   "request": {
	//     "$ref": "RunQueryRequest"
	//   },
	//   "response": {
	//     "$ref": "RunQueryResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/datastore",
	//     "https://www.googleapis.com/auth/userinfo.email"
	//   ]
	// }

}
