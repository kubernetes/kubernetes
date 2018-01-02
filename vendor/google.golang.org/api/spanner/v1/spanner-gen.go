// Package spanner provides access to the Cloud Spanner API.
//
// See https://cloud.google.com/spanner/
//
// Usage example:
//
//   import "google.golang.org/api/spanner/v1"
//   ...
//   spannerService, err := spanner.New(oauthHttpClient)
package spanner // import "google.golang.org/api/spanner/v1"

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

const apiId = "spanner:v1"
const apiName = "spanner"
const apiVersion = "v1"
const basePath = "https://spanner.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// Administer your Spanner databases
	SpannerAdminScope = "https://www.googleapis.com/auth/spanner.admin"

	// View and manage the contents of your Spanner databases
	SpannerDataScope = "https://www.googleapis.com/auth/spanner.data"
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
	rs.InstanceConfigs = NewProjectsInstanceConfigsService(s)
	rs.Instances = NewProjectsInstancesService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	InstanceConfigs *ProjectsInstanceConfigsService

	Instances *ProjectsInstancesService
}

func NewProjectsInstanceConfigsService(s *Service) *ProjectsInstanceConfigsService {
	rs := &ProjectsInstanceConfigsService{s: s}
	return rs
}

type ProjectsInstanceConfigsService struct {
	s *Service
}

func NewProjectsInstancesService(s *Service) *ProjectsInstancesService {
	rs := &ProjectsInstancesService{s: s}
	rs.Databases = NewProjectsInstancesDatabasesService(s)
	rs.Operations = NewProjectsInstancesOperationsService(s)
	return rs
}

type ProjectsInstancesService struct {
	s *Service

	Databases *ProjectsInstancesDatabasesService

	Operations *ProjectsInstancesOperationsService
}

func NewProjectsInstancesDatabasesService(s *Service) *ProjectsInstancesDatabasesService {
	rs := &ProjectsInstancesDatabasesService{s: s}
	rs.Operations = NewProjectsInstancesDatabasesOperationsService(s)
	rs.Sessions = NewProjectsInstancesDatabasesSessionsService(s)
	return rs
}

type ProjectsInstancesDatabasesService struct {
	s *Service

	Operations *ProjectsInstancesDatabasesOperationsService

	Sessions *ProjectsInstancesDatabasesSessionsService
}

func NewProjectsInstancesDatabasesOperationsService(s *Service) *ProjectsInstancesDatabasesOperationsService {
	rs := &ProjectsInstancesDatabasesOperationsService{s: s}
	return rs
}

type ProjectsInstancesDatabasesOperationsService struct {
	s *Service
}

func NewProjectsInstancesDatabasesSessionsService(s *Service) *ProjectsInstancesDatabasesSessionsService {
	rs := &ProjectsInstancesDatabasesSessionsService{s: s}
	return rs
}

type ProjectsInstancesDatabasesSessionsService struct {
	s *Service
}

func NewProjectsInstancesOperationsService(s *Service) *ProjectsInstancesOperationsService {
	rs := &ProjectsInstancesOperationsService{s: s}
	return rs
}

type ProjectsInstancesOperationsService struct {
	s *Service
}

// BeginTransactionRequest: The request for BeginTransaction.
type BeginTransactionRequest struct {
	// Options: Required. Options for the new transaction.
	Options *TransactionOptions `json:"options,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Options") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Options") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BeginTransactionRequest) MarshalJSON() ([]byte, error) {
	type noMethod BeginTransactionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Binding: Associates `members` with a `role`.
type Binding struct {
	// Members: Specifies the identities requesting access for a Cloud
	// Platform resource.
	// `members` can have the following values:
	//
	// * `allUsers`: A special identifier that represents anyone who is
	//    on the internet; with or without a Google account.
	//
	// * `allAuthenticatedUsers`: A special identifier that represents
	// anyone
	//    who is authenticated with a Google account or a service
	// account.
	//
	// * `user:{emailid}`: An email address that represents a specific
	// Google
	//    account. For example, `alice@gmail.com` or `joe@example.com`.
	//
	//
	// * `serviceAccount:{emailid}`: An email address that represents a
	// service
	//    account. For example,
	// `my-other-app@appspot.gserviceaccount.com`.
	//
	// * `group:{emailid}`: An email address that represents a Google
	// group.
	//    For example, `admins@example.com`.
	//
	//
	// * `domain:{domain}`: A Google Apps domain name that represents all
	// the
	//    users of that domain. For example, `google.com` or
	// `example.com`.
	//
	//
	Members []string `json:"members,omitempty"`

	// Role: Role that is assigned to `members`.
	// For example, `roles/viewer`, `roles/editor`, or
	// `roles/owner`.
	// Required
	Role string `json:"role,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Members") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Members") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Binding) MarshalJSON() ([]byte, error) {
	type noMethod Binding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ChildLink: Metadata associated with a parent-child relationship
// appearing in a
// PlanNode.
type ChildLink struct {
	// ChildIndex: The node to which the link points.
	ChildIndex int64 `json:"childIndex,omitempty"`

	// Type: The type of the link. For example, in Hash Joins this could be
	// used to
	// distinguish between the build child and the probe child, or in the
	// case
	// of the child being an output variable, to represent the tag
	// associated
	// with the output variable.
	Type string `json:"type,omitempty"`

	// Variable: Only present if the child node is SCALAR and corresponds
	// to an output variable of the parent node. The field carries the name
	// of
	// the output variable.
	// For example, a `TableScan` operator that reads rows from a table
	// will
	// have child links to the `SCALAR` nodes representing the output
	// variables
	// created for each column that is read by the operator. The
	// corresponding
	// `variable` fields will be set to the variable names assigned to
	// the
	// columns.
	Variable string `json:"variable,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ChildIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChildIndex") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ChildLink) MarshalJSON() ([]byte, error) {
	type noMethod ChildLink
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommitRequest: The request for Commit.
type CommitRequest struct {
	// Mutations: The mutations to be executed when this transaction
	// commits. All
	// mutations are applied atomically, in the order they appear in
	// this list.
	Mutations []*Mutation `json:"mutations,omitempty"`

	// SingleUseTransaction: Execute mutations in a temporary transaction.
	// Note that unlike
	// commit of a previously-started transaction, commit with a
	// temporary transaction is non-idempotent. That is, if
	// the
	// `CommitRequest` is sent to Cloud Spanner more than once
	// (for
	// instance, due to retries in the application, or in the
	// transport library), it is possible that the mutations are
	// executed more than once. If this is undesirable, use
	// BeginTransaction and
	// Commit instead.
	SingleUseTransaction *TransactionOptions `json:"singleUseTransaction,omitempty"`

	// TransactionId: Commit a previously-started transaction.
	TransactionId string `json:"transactionId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Mutations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Mutations") to include in
	// API requests with the JSON null value. By default, fields with empty
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

// CommitResponse: The response for Commit.
type CommitResponse struct {
	// CommitTimestamp: The Cloud Spanner timestamp at which the transaction
	// committed.
	CommitTimestamp string `json:"commitTimestamp,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CommitTimestamp") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CommitTimestamp") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CommitResponse) MarshalJSON() ([]byte, error) {
	type noMethod CommitResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateDatabaseMetadata: Metadata type for the operation returned
// by
// CreateDatabase.
type CreateDatabaseMetadata struct {
	// Database: The database being created.
	Database string `json:"database,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Database") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Database") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateDatabaseMetadata) MarshalJSON() ([]byte, error) {
	type noMethod CreateDatabaseMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateDatabaseRequest: The request for CreateDatabase.
type CreateDatabaseRequest struct {
	// CreateStatement: Required. A `CREATE DATABASE` statement, which
	// specifies the ID of the
	// new database.  The database ID must conform to the regular
	// expression
	// `a-z*[a-z0-9]` and be between 2 and 30 characters in length.
	// If the database ID is a reserved word or if it contains a hyphen,
	// the
	// database ID must be enclosed in backticks (`` ` ``).
	CreateStatement string `json:"createStatement,omitempty"`

	// ExtraStatements: An optional list of DDL statements to run inside the
	// newly created
	// database. Statements can create tables, indexes, etc.
	// These
	// statements execute atomically with the creation of the database:
	// if there is an error in any statement, the database is not created.
	ExtraStatements []string `json:"extraStatements,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreateStatement") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateStatement") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CreateDatabaseRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateDatabaseRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateInstanceMetadata: Metadata type for the operation returned
// by
// CreateInstance.
type CreateInstanceMetadata struct {
	// CancelTime: The time at which this operation was cancelled. If set,
	// this operation is
	// in the process of undoing itself (which is guaranteed to succeed)
	// and
	// cannot be cancelled again.
	CancelTime string `json:"cancelTime,omitempty"`

	// EndTime: The time at which this operation failed or was completed
	// successfully.
	EndTime string `json:"endTime,omitempty"`

	// Instance: The instance being created.
	Instance *Instance `json:"instance,omitempty"`

	// StartTime: The time at which the
	// CreateInstance request was
	// received.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CancelTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CancelTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateInstanceMetadata) MarshalJSON() ([]byte, error) {
	type noMethod CreateInstanceMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateInstanceRequest: The request for CreateInstance.
type CreateInstanceRequest struct {
	// Instance: Required. The instance to create.  The name may be omitted,
	// but if
	// specified must be `<parent>/instances/<instance_id>`.
	Instance *Instance `json:"instance,omitempty"`

	// InstanceId: Required. The ID of the instance to create.  Valid
	// identifiers are of the
	// form `a-z*[a-z0-9]` and must be between 6 and 30 characters
	// in
	// length.
	InstanceId string `json:"instanceId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Instance") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Instance") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateInstanceRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateInstanceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Database: A Cloud Spanner database.
type Database struct {
	// Name: Required. The name of the database. Values are of the
	// form
	// `projects/<project>/instances/<instance>/databases/<database>`,
	// w
	// here `<database>` is as specified in the `CREATE DATABASE`
	// statement. This name can be passed to other API methods to
	// identify the database.
	Name string `json:"name,omitempty"`

	// State: Output only. The current database state.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED" - Not specified.
	//   "CREATING" - The database is still being created. Operations on the
	// database may fail
	// with `FAILED_PRECONDITION` in this state.
	//   "READY" - The database is fully created and ready for use.
	State string `json:"state,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *Database) MarshalJSON() ([]byte, error) {
	type noMethod Database
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Delete: Arguments to delete operations.
type Delete struct {
	// KeySet: Required. The primary keys of the rows within table to
	// delete.
	KeySet *KeySet `json:"keySet,omitempty"`

	// Table: Required. The table whose rows will be deleted.
	Table string `json:"table,omitempty"`

	// ForceSendFields is a list of field names (e.g. "KeySet") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "KeySet") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Delete) MarshalJSON() ([]byte, error) {
	type noMethod Delete
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

// ExecuteSqlRequest: The request for ExecuteSql
// and
// ExecuteStreamingSql.
type ExecuteSqlRequest struct {
	// ParamTypes: It is not always possible for Cloud Spanner to infer the
	// right SQL type
	// from a JSON value.  For example, values of type `BYTES` and values
	// of type `STRING` both appear in params as JSON strings.
	//
	// In these cases, `param_types` can be used to specify the exact
	// SQL type for some or all of the SQL query parameters. See
	// the
	// definition of Type for more information
	// about SQL types.
	ParamTypes map[string]Type `json:"paramTypes,omitempty"`

	// Params: The SQL query string can contain parameter placeholders. A
	// parameter
	// placeholder consists of `'@'` followed by the parameter
	// name. Parameter names consist of any combination of letters,
	// numbers, and underscores.
	//
	// Parameters can appear anywhere that a literal value is expected.  The
	// same
	// parameter name can be used more than once, for example:
	//   "WHERE id > @msg_id AND id < @msg_id + 100"
	//
	// It is an error to execute an SQL query with unbound
	// parameters.
	//
	// Parameter values are specified using `params`, which is a JSON
	// object whose keys are parameter names, and whose values are
	// the
	// corresponding parameter values.
	Params googleapi.RawMessage `json:"params,omitempty"`

	// QueryMode: Used to control the amount of debugging information
	// returned in
	// ResultSetStats.
	//
	// Possible values:
	//   "NORMAL" - The default mode where only the query result, without
	// any information
	// about the query plan is returned.
	//   "PLAN" - This mode returns only the query plan, without any result
	// rows or
	// execution statistics information.
	//   "PROFILE" - This mode returns both the query plan and the execution
	// statistics along
	// with the result rows.
	QueryMode string `json:"queryMode,omitempty"`

	// ResumeToken: If this request is resuming a previously interrupted SQL
	// query
	// execution, `resume_token` should be copied from the
	// last
	// PartialResultSet yielded before the interruption. Doing this
	// enables the new SQL query execution to resume where the last one
	// left
	// off. The rest of the request parameters must exactly match
	// the
	// request that yielded this token.
	ResumeToken string `json:"resumeToken,omitempty"`

	// Sql: Required. The SQL query string.
	Sql string `json:"sql,omitempty"`

	// Transaction: The transaction to use. If none is provided, the default
	// is a
	// temporary read-only transaction with strong concurrency.
	Transaction *TransactionSelector `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ParamTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ParamTypes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExecuteSqlRequest) MarshalJSON() ([]byte, error) {
	type noMethod ExecuteSqlRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Field: Message representing a single field of a struct.
type Field struct {
	// Name: The name of the field. For reads, this is the column name.
	// For
	// SQL queries, it is the column alias (e.g., "Word" in the
	// query "SELECT 'hello' AS Word"), or the column name
	// (e.g.,
	// "ColName" in the query "SELECT ColName FROM Table"). Some
	// columns might have an empty name (e.g., !"SELECT
	// UPPER(ColName)"). Note that a query result can contain
	// multiple fields with the same name.
	Name string `json:"name,omitempty"`

	// Type: The type of the field.
	Type *Type `json:"type,omitempty"`

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

func (s *Field) MarshalJSON() ([]byte, error) {
	type noMethod Field
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetDatabaseDdlResponse: The response for GetDatabaseDdl.
type GetDatabaseDdlResponse struct {
	// Statements: A list of formatted DDL statements defining the schema of
	// the database
	// specified in the request.
	Statements []string `json:"statements,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Statements") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Statements") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GetDatabaseDdlResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetDatabaseDdlResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetIamPolicyRequest: Request message for `GetIamPolicy` method.
type GetIamPolicyRequest struct {
}

// Instance: An isolated set of Cloud Spanner resources on which
// databases can be hosted.
type Instance struct {
	// Config: Required. The name of the instance's configuration. Values
	// are of the form
	// `projects/<project>/instanceConfigs/<configuration>`. See
	// also InstanceConfig and
	// ListInstanceConfigs.
	Config string `json:"config,omitempty"`

	// DisplayName: Required. The descriptive name for this instance as it
	// appears in UIs.
	// Must be unique per project and between 4 and 30 characters in length.
	DisplayName string `json:"displayName,omitempty"`

	// Labels: Cloud Labels are a flexible and lightweight mechanism for
	// organizing cloud
	// resources into groups that reflect a customer's organizational needs
	// and
	// deployment strategies. Cloud Labels can be used to filter collections
	// of
	// resources. They can be used to control how resource metrics are
	// aggregated.
	// And they can be used as arguments to policy management rules (e.g.
	// route,
	// firewall, load balancing, etc.).
	//
	//  * Label keys must be between 1 and 63 characters long and must
	// conform to
	//    the following regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`.
	//  * Label values must be between 0 and 63 characters long and must
	// conform
	//    to the regular expression `([a-z]([-a-z0-9]*[a-z0-9])?)?`.
	//  * No more than 64 labels can be associated with a given
	// resource.
	//
	// See https://goo.gl/xmQnxf for more information on and examples of
	// labels.
	//
	// If you plan to use labels in your own code, please note that
	// additional
	// characters may be allowed in the future. And so you are advised to
	// use an
	// internal label representation, such as JSON, which doesn't rely
	// upon
	// specific characters being disallowed.  For example, representing
	// labels
	// as the string:  name + "_" + value  would prove problematic if we
	// were to
	// allow "_" in a future release.
	Labels map[string]string `json:"labels,omitempty"`

	// Name: Required. A unique identifier for the instance, which cannot be
	// changed
	// after the instance is created. Values are of the
	// form
	// `projects/<project>/instances/a-z*[a-z0-9]`. The final
	// segment of the name must be between 6 and 30 characters in length.
	Name string `json:"name,omitempty"`

	// NodeCount: Required. The number of nodes allocated to this instance.
	// This may be zero
	// in API responses for instances that are not yet in state
	// `READY`.
	//
	// Each Spanner node can provide up to 10,000 QPS of reads or 2000 QPS
	// of
	// writes (writing single rows at 1KB data per row), and 2 TiB
	// storage.
	//
	// For optimal performance, we recommend provisioning enough nodes to
	// keep
	// overall CPU utilization under 75%.
	//
	// A minimum of 3 nodes is recommended for production environments.
	// This
	// minimum is required for SLAs to apply to your instance.
	//
	// Note that Cloud Spanner performance is highly dependent on workload,
	// schema
	// design, and dataset characteristics. The performance numbers above
	// are
	// estimates, and assume [best
	// practices](https://cloud.google.com/spanner/docs/bulk-loading)
	// are followed.
	NodeCount int64 `json:"nodeCount,omitempty"`

	// State: Output only. The current instance state. For
	// CreateInstance, the state must be
	// either omitted or set to `CREATING`. For
	// UpdateInstance, the state must be
	// either omitted or set to `READY`.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED" - Not specified.
	//   "CREATING" - The instance is still being created. Resources may not
	// be
	// available yet, and operations such as database creation may not
	// work.
	//   "READY" - The instance is fully created and ready to do work such
	// as
	// creating databases.
	State string `json:"state,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Config") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Config") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Instance) MarshalJSON() ([]byte, error) {
	type noMethod Instance
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstanceConfig: A possible configuration for a Cloud Spanner
// instance. Configurations
// define the geographic placement of nodes and their replication.
type InstanceConfig struct {
	// DisplayName: The name of this instance configuration as it appears in
	// UIs.
	DisplayName string `json:"displayName,omitempty"`

	// Name: A unique identifier for the instance configuration.  Values
	// are of the form
	// `projects/<project>/instanceConfigs/a-z*`
	Name string `json:"name,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *InstanceConfig) MarshalJSON() ([]byte, error) {
	type noMethod InstanceConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// KeyRange: KeyRange represents a range of rows in a table or index.
//
// A range has a start key and an end key. These keys can be open
// or
// closed, indicating if the range includes rows with that key.
//
// Keys are represented by lists, where the ith value in the
// list
// corresponds to the ith component of the table or index primary
// key.
// Individual values are encoded as described here.
//
// For example, consider the following table definition:
//
//     CREATE TABLE UserEvents (
//       UserName STRING(MAX),
//       EventDate STRING(10)
//     ) PRIMARY KEY(UserName, EventDate);
//
// The following keys name rows in this table:
//
//     "Bob", "2014-09-23"
//
// Since the `UserEvents` table's `PRIMARY KEY` clause names
// two
// columns, each `UserEvents` key has two elements; the first is
// the
// `UserName`, and the second is the `EventDate`.
//
// Key ranges with multiple components are interpreted
// lexicographically by component using the table or index key's
// declared
// sort order. For example, the following range returns all events
// for
// user "Bob" that occurred in the year 2015:
//
//     "start_closed": ["Bob", "2015-01-01"]
//     "end_closed": ["Bob", "2015-12-31"]
//
// Start and end keys can omit trailing key components. This affects
// the
// inclusion and exclusion of rows that exactly match the provided
// key
// components: if the key is closed, then rows that exactly match
// the
// provided components are included; if the key is open, then rows
// that exactly match are not included.
//
// For example, the following range includes all events for "Bob"
// that
// occurred during and after the year 2000:
//
//     "start_closed": ["Bob", "2000-01-01"]
//     "end_closed": ["Bob"]
//
// The next example retrieves all events for "Bob":
//
//     "start_closed": ["Bob"]
//     "end_closed": ["Bob"]
//
// To retrieve events before the year 2000:
//
//     "start_closed": ["Bob"]
//     "end_open": ["Bob", "2000-01-01"]
//
// The following range includes all rows in the table:
//
//     "start_closed": []
//     "end_closed": []
//
// This range returns all users whose `UserName` begins with
// any
// character from A to C:
//
//     "start_closed": ["A"]
//     "end_open": ["D"]
//
// This range returns all users whose `UserName` begins with B:
//
//     "start_closed": ["B"]
//     "end_open": ["C"]
//
// Key ranges honor column sort order. For example, suppose a table
// is
// defined as follows:
//
//     CREATE TABLE DescendingSortedTable {
//       Key INT64,
//       ...
//     ) PRIMARY KEY(Key DESC);
//
// The following range retrieves all rows with key values between 1
// and 100 inclusive:
//
//     "start_closed": ["100"]
//     "end_closed": ["1"]
//
// Note that 100 is passed as the start, and 1 is passed as the
// end,
// because `Key` is a descending column in the schema.
type KeyRange struct {
	// EndClosed: If the end is closed, then the range includes all rows
	// whose
	// first `len(end_closed)` key columns exactly match `end_closed`.
	EndClosed []interface{} `json:"endClosed,omitempty"`

	// EndOpen: If the end is open, then the range excludes rows whose
	// first
	// `len(end_open)` key columns exactly match `end_open`.
	EndOpen []interface{} `json:"endOpen,omitempty"`

	// StartClosed: If the start is closed, then the range includes all rows
	// whose
	// first `len(start_closed)` key columns exactly match `start_closed`.
	StartClosed []interface{} `json:"startClosed,omitempty"`

	// StartOpen: If the start is open, then the range excludes rows whose
	// first
	// `len(start_open)` key columns exactly match `start_open`.
	StartOpen []interface{} `json:"startOpen,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndClosed") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndClosed") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *KeyRange) MarshalJSON() ([]byte, error) {
	type noMethod KeyRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// KeySet: `KeySet` defines a collection of Cloud Spanner keys and/or
// key ranges. All
// the keys are expected to be in the same table or index. The keys
// need
// not be sorted in any particular way.
//
// If the same key is specified multiple times in the set (for
// example
// if two ranges, two keys, or a key and a range overlap), Cloud
// Spanner
// behaves as if the key were only specified once.
type KeySet struct {
	// All: For convenience `all` can be set to `true` to indicate that
	// this
	// `KeySet` matches all keys in the table or index. Note that any
	// keys
	// specified in `keys` or `ranges` are only yielded once.
	All bool `json:"all,omitempty"`

	// Keys: A list of specific keys. Entries in `keys` should have exactly
	// as
	// many elements as there are columns in the primary or index key
	// with which this `KeySet` is used.  Individual key values are
	// encoded as described here.
	Keys [][]interface{} `json:"keys,omitempty"`

	// Ranges: A list of key ranges. See KeyRange for more information
	// about
	// key range specifications.
	Ranges []*KeyRange `json:"ranges,omitempty"`

	// ForceSendFields is a list of field names (e.g. "All") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "All") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *KeySet) MarshalJSON() ([]byte, error) {
	type noMethod KeySet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListDatabasesResponse: The response for ListDatabases.
type ListDatabasesResponse struct {
	// Databases: Databases that matched the request.
	Databases []*Database `json:"databases,omitempty"`

	// NextPageToken: `next_page_token` can be sent in a
	// subsequent
	// ListDatabases call to fetch more
	// of the matching databases.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Databases") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Databases") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListDatabasesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListDatabasesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListInstanceConfigsResponse: The response for ListInstanceConfigs.
type ListInstanceConfigsResponse struct {
	// InstanceConfigs: The list of requested instance configurations.
	InstanceConfigs []*InstanceConfig `json:"instanceConfigs,omitempty"`

	// NextPageToken: `next_page_token` can be sent in a
	// subsequent
	// ListInstanceConfigs call to
	// fetch more of the matching instance configurations.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "InstanceConfigs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InstanceConfigs") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListInstanceConfigsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListInstanceConfigsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListInstancesResponse: The response for ListInstances.
type ListInstancesResponse struct {
	// Instances: The list of requested instances.
	Instances []*Instance `json:"instances,omitempty"`

	// NextPageToken: `next_page_token` can be sent in a
	// subsequent
	// ListInstances call to fetch more
	// of the matching instances.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Instances") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Instances") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListInstancesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListInstancesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListOperationsResponse: The response message for
// Operations.ListOperations.
type ListOperationsResponse struct {
	// NextPageToken: The standard List next-page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Operations: A list of operations that matches the specified filter in
	// the request.
	Operations []*Operation `json:"operations,omitempty"`

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

func (s *ListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Mutation: A modification to one or more Cloud Spanner rows.
// Mutations can be
// applied to a Cloud Spanner database by sending them in a
// Commit call.
type Mutation struct {
	// Delete: Delete rows from a table. Succeeds whether or not the
	// named
	// rows were present.
	Delete *Delete `json:"delete,omitempty"`

	// Insert: Insert new rows in a table. If any of the rows already
	// exist,
	// the write or transaction fails with error `ALREADY_EXISTS`.
	Insert *Write `json:"insert,omitempty"`

	// InsertOrUpdate: Like insert, except that if the row already exists,
	// then
	// its column values are overwritten with the ones provided. Any
	// column values not explicitly written are preserved.
	InsertOrUpdate *Write `json:"insertOrUpdate,omitempty"`

	// Replace: Like insert, except that if the row already exists, it
	// is
	// deleted, and the column values provided are inserted
	// instead. Unlike insert_or_update, this means any values
	// not
	// explicitly written become `NULL`.
	Replace *Write `json:"replace,omitempty"`

	// Update: Update existing rows in a table. If any of the rows does
	// not
	// already exist, the transaction fails with error `NOT_FOUND`.
	Update *Write `json:"update,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Delete") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Delete") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Mutation) MarshalJSON() ([]byte, error) {
	type noMethod Mutation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: This resource represents a long-running operation that is
// the result of a
// network API call.
type Operation struct {
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

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PartialResultSet: Partial results from a streaming read or SQL query.
// Streaming reads and
// SQL queries better tolerate large result sets, large rows, and
// large
// values, but are a little trickier to consume.
type PartialResultSet struct {
	// ChunkedValue: If true, then the final value in values is chunked, and
	// must
	// be combined with more values from subsequent `PartialResultSet`s
	// to obtain a complete field value.
	ChunkedValue bool `json:"chunkedValue,omitempty"`

	// Metadata: Metadata about the result set, such as row type
	// information.
	// Only present in the first response.
	Metadata *ResultSetMetadata `json:"metadata,omitempty"`

	// ResumeToken: Streaming calls might be interrupted for a variety of
	// reasons, such
	// as TCP connection loss. If this occurs, the stream of results can
	// be resumed by re-sending the original request and
	// including
	// `resume_token`. Note that executing any other transaction in the
	// same session invalidates the token.
	ResumeToken string `json:"resumeToken,omitempty"`

	// Stats: Query plan and execution statistics for the query that
	// produced this
	// streaming result set. These can be requested by
	// setting
	// ExecuteSqlRequest.query_mode and are sent
	// only once with the last response in the stream.
	Stats *ResultSetStats `json:"stats,omitempty"`

	// Values: A streamed result set consists of a stream of values, which
	// might
	// be split into many `PartialResultSet` messages to accommodate
	// large rows and/or large values. Every N complete values defines
	// a
	// row, where N is equal to the number of entries
	// in
	// metadata.row_type.fields.
	//
	// Most values are encoded based on type as described
	// here.
	//
	// It is possible that the last value in values is "chunked",
	// meaning that the rest of the value is sent in
	// subsequent
	// `PartialResultSet`(s). This is denoted by the chunked_value
	// field. Two or more chunked values can be merged to form a
	// complete value as follows:
	//
	//   * `bool/number/null`: cannot be chunked
	//   * `string`: concatenate the strings
	//   * `list`: concatenate the lists. If the last element in a list is
	// a
	//     `string`, `list`, or `object`, merge it with the first element
	// in
	//     the next list by applying these rules recursively.
	//   * `object`: concatenate the (field name, field value) pairs. If a
	//     field name is duplicated, then apply these rules recursively
	//     to merge the field values.
	//
	// Some examples of merging:
	//
	//     # Strings are concatenated.
	//     "foo", "bar" => "foobar"
	//
	//     # Lists of non-strings are concatenated.
	//     [2, 3], [4] => [2, 3, 4]
	//
	//     # Lists are concatenated, but the last and first elements are
	// merged
	//     # because they are strings.
	//     ["a", "b"], ["c", "d"] => ["a", "bc", "d"]
	//
	//     # Lists are concatenated, but the last and first elements are
	// merged
	//     # because they are lists. Recursively, the last and first
	// elements
	//     # of the inner lists are merged because they are strings.
	//     ["a", ["b", "c"]], [["d"], "e"] => ["a", ["b", "cd"], "e"]
	//
	//     # Non-overlapping object fields are combined.
	//     {"a": "1"}, {"b": "2"} => {"a": "1", "b": 2"}
	//
	//     # Overlapping object fields are merged.
	//     {"a": "1"}, {"a": "2"} => {"a": "12"}
	//
	//     # Examples of merging objects containing lists of strings.
	//     {"a": ["1"]}, {"a": ["2"]} => {"a": ["12"]}
	//
	// For a more complete example, suppose a streaming SQL query
	// is
	// yielding a result set whose rows contain a single string
	// field. The following `PartialResultSet`s might be yielded:
	//
	//     {
	//       "metadata": { ... }
	//       "values": ["Hello", "W"]
	//       "chunked_value": true
	//       "resume_token": "Af65..."
	//     }
	//     {
	//       "values": ["orl"]
	//       "chunked_value": true
	//       "resume_token": "Bqp2..."
	//     }
	//     {
	//       "values": ["d"]
	//       "resume_token": "Zx1B..."
	//     }
	//
	// This sequence of `PartialResultSet`s encodes two rows, one
	// containing the field value "Hello", and a second containing
	// the
	// field value "World" = "W" + "orl" + "d".
	Values []interface{} `json:"values,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ChunkedValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChunkedValue") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PartialResultSet) MarshalJSON() ([]byte, error) {
	type noMethod PartialResultSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PlanNode: Node information for nodes appearing in a
// QueryPlan.plan_nodes.
type PlanNode struct {
	// ChildLinks: List of child node `index`es and their relationship to
	// this parent.
	ChildLinks []*ChildLink `json:"childLinks,omitempty"`

	// DisplayName: The display name for the node.
	DisplayName string `json:"displayName,omitempty"`

	// ExecutionStats: The execution statistics associated with the node,
	// contained in a group of
	// key-value pairs. Only present if the plan was returned as a result of
	// a
	// profile query. For example, number of executions, number of rows/time
	// per
	// execution etc.
	ExecutionStats googleapi.RawMessage `json:"executionStats,omitempty"`

	// Index: The `PlanNode`'s index in node list.
	Index int64 `json:"index,omitempty"`

	// Kind: Used to determine the type of node. May be needed for
	// visualizing
	// different kinds of nodes differently. For example, If the node is
	// a
	// SCALAR node, it will have a condensed representation
	// which can be used to directly embed a description of the node in
	// its
	// parent.
	//
	// Possible values:
	//   "KIND_UNSPECIFIED" - Not specified.
	//   "RELATIONAL" - Denotes a Relational operator node in the expression
	// tree. Relational
	// operators represent iterative processing of rows during query
	// execution.
	// For example, a `TableScan` operation that reads rows from a table.
	//   "SCALAR" - Denotes a Scalar node in the expression tree. Scalar
	// nodes represent
	// non-iterable entities in the query plan. For example, constants
	// or
	// arithmetic operators appearing inside predicate expressions or
	// references
	// to column names.
	Kind string `json:"kind,omitempty"`

	// Metadata: Attributes relevant to the node contained in a group of
	// key-value pairs.
	// For example, a Parameter Reference node could have the
	// following
	// information in its metadata:
	//
	//     {
	//       "parameter_reference": "param1",
	//       "parameter_type": "array"
	//     }
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// ShortRepresentation: Condensed representation for SCALAR nodes.
	ShortRepresentation *ShortRepresentation `json:"shortRepresentation,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ChildLinks") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChildLinks") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PlanNode) MarshalJSON() ([]byte, error) {
	type noMethod PlanNode
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Policy: Defines an Identity and Access Management (IAM) policy. It is
// used to
// specify access control policies for Cloud Platform resources.
//
//
// A `Policy` consists of a list of `bindings`. A `Binding` binds a list
// of
// `members` to a `role`, where the members can be user accounts, Google
// groups,
// Google domains, and service accounts. A `role` is a named list of
// permissions
// defined by IAM.
//
// **Example**
//
//     {
//       "bindings": [
//         {
//           "role": "roles/owner",
//           "members": [
//             "user:mike@example.com",
//             "group:admins@example.com",
//             "domain:google.com",
//
// "serviceAccount:my-other-app@appspot.gserviceaccount.com",
//           ]
//         },
//         {
//           "role": "roles/viewer",
//           "members": ["user:sean@example.com"]
//         }
//       ]
//     }
//
// For a description of IAM and its features, see the
// [IAM developer's guide](https://cloud.google.com/iam).
type Policy struct {
	// Bindings: Associates a list of `members` to a `role`.
	// `bindings` with no members will result in an error.
	Bindings []*Binding `json:"bindings,omitempty"`

	// Etag: `etag` is used for optimistic concurrency control as a way to
	// help
	// prevent simultaneous updates of a policy from overwriting each
	// other.
	// It is strongly suggested that systems make use of the `etag` in
	// the
	// read-modify-write cycle to perform policy updates in order to avoid
	// race
	// conditions: An `etag` is returned in the response to `getIamPolicy`,
	// and
	// systems are expected to put that etag in the request to
	// `setIamPolicy` to
	// ensure that their change will be applied to the same version of the
	// policy.
	//
	// If no `etag` is provided in the call to `setIamPolicy`, then the
	// existing
	// policy is overwritten blindly.
	Etag string `json:"etag,omitempty"`

	// Version: Version of the `Policy`. The default version is 0.
	Version int64 `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Bindings") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bindings") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Policy) MarshalJSON() ([]byte, error) {
	type noMethod Policy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QueryPlan: Contains an ordered list of nodes appearing in the query
// plan.
type QueryPlan struct {
	// PlanNodes: The nodes in the query plan. Plan nodes are returned in
	// pre-order starting
	// with the plan root. Each PlanNode's `id` corresponds to its index
	// in
	// `plan_nodes`.
	PlanNodes []*PlanNode `json:"planNodes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PlanNodes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PlanNodes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryPlan) MarshalJSON() ([]byte, error) {
	type noMethod QueryPlan
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReadOnly: Message type to initiate a read-only transaction.
type ReadOnly struct {
	// ExactStaleness: Executes all reads at a timestamp that is
	// `exact_staleness`
	// old. The timestamp is chosen soon after the read is
	// started.
	//
	// Guarantees that all writes that have committed more than
	// the
	// specified number of seconds ago are visible. Because Cloud
	// Spanner
	// chooses the exact timestamp, this mode works even if the
	// client's
	// local clock is substantially skewed from Cloud Spanner
	// commit
	// timestamps.
	//
	// Useful for reading at nearby replicas without the
	// distributed
	// timestamp negotiation overhead of `max_staleness`.
	ExactStaleness string `json:"exactStaleness,omitempty"`

	// MaxStaleness: Read data at a timestamp >= `NOW -
	// max_staleness`
	// seconds. Guarantees that all writes that have committed more
	// than the specified number of seconds ago are visible. Because
	// Cloud Spanner chooses the exact timestamp, this mode works even
	// if
	// the client's local clock is substantially skewed from Cloud
	// Spanner
	// commit timestamps.
	//
	// Useful for reading the freshest data available at a nearby
	// replica, while bounding the possible staleness if the local
	// replica has fallen behind.
	//
	// Note that this option can only be used in single-use
	// transactions.
	MaxStaleness string `json:"maxStaleness,omitempty"`

	// MinReadTimestamp: Executes all reads at a timestamp >=
	// `min_read_timestamp`.
	//
	// This is useful for requesting fresher data than some previous
	// read, or data that is fresh enough to observe the effects of
	// some
	// previously committed transaction whose timestamp is known.
	//
	// Note that this option can only be used in single-use transactions.
	MinReadTimestamp string `json:"minReadTimestamp,omitempty"`

	// ReadTimestamp: Executes all reads at the given timestamp. Unlike
	// other modes,
	// reads at a specific timestamp are repeatable; the same read at
	// the same timestamp always returns the same data. If the
	// timestamp is in the future, the read will block until the
	// specified timestamp, modulo the read's deadline.
	//
	// Useful for large scale consistent reads such as mapreduces, or
	// for coordinating many reads against a consistent snapshot of
	// the
	// data.
	ReadTimestamp string `json:"readTimestamp,omitempty"`

	// ReturnReadTimestamp: If true, the Cloud Spanner-selected read
	// timestamp is included in
	// the Transaction message that describes the transaction.
	ReturnReadTimestamp bool `json:"returnReadTimestamp,omitempty"`

	// Strong: Read at a timestamp where all previously committed
	// transactions
	// are visible.
	Strong bool `json:"strong,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExactStaleness") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExactStaleness") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ReadOnly) MarshalJSON() ([]byte, error) {
	type noMethod ReadOnly
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReadRequest: The request for Read and
// StreamingRead.
type ReadRequest struct {
	// Columns: The columns of table to be returned for each row
	// matching
	// this request.
	Columns []string `json:"columns,omitempty"`

	// Index: If non-empty, the name of an index on table. This index
	// is
	// used instead of the table primary key when interpreting key_set
	// and sorting result rows. See key_set for further information.
	Index string `json:"index,omitempty"`

	// KeySet: Required. `key_set` identifies the rows to be yielded.
	// `key_set` names the
	// primary keys of the rows in table to be yielded, unless index
	// is present. If index is present, then key_set instead names
	// index keys in index.
	//
	// Rows are yielded in table primary key order (if index is empty)
	// or index key order (if index is non-empty).
	//
	// It is not an error for the `key_set` to name rows that do not
	// exist in the database. Read yields nothing for nonexistent rows.
	KeySet *KeySet `json:"keySet,omitempty"`

	// Limit: If greater than zero, only the first `limit` rows are yielded.
	// If `limit`
	// is zero, the default is no limit.
	Limit int64 `json:"limit,omitempty,string"`

	// ResumeToken: If this request is resuming a previously interrupted
	// read,
	// `resume_token` should be copied from the last
	// PartialResultSet yielded before the interruption. Doing this
	// enables the new read to resume where the last read left off. The
	// rest of the request parameters must exactly match the request
	// that yielded this token.
	ResumeToken string `json:"resumeToken,omitempty"`

	// Table: Required. The name of the table in the database to be read.
	Table string `json:"table,omitempty"`

	// Transaction: The transaction to use. If none is provided, the default
	// is a
	// temporary read-only transaction with strong concurrency.
	Transaction *TransactionSelector `json:"transaction,omitempty"`

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

func (s *ReadRequest) MarshalJSON() ([]byte, error) {
	type noMethod ReadRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReadWrite: Message type to initiate a read-write transaction.
// Currently this
// transaction type has no options.
type ReadWrite struct {
}

// ResultSet: Results from Read or
// ExecuteSql.
type ResultSet struct {
	// Metadata: Metadata about the result set, such as row type
	// information.
	Metadata *ResultSetMetadata `json:"metadata,omitempty"`

	// Rows: Each element in `rows` is a row whose format is defined
	// by
	// metadata.row_type. The ith element
	// in each row matches the ith field in
	// metadata.row_type. Elements are
	// encoded based on type as described
	// here.
	Rows [][]interface{} `json:"rows,omitempty"`

	// Stats: Query plan and execution statistics for the query that
	// produced this
	// result set. These can be requested by
	// setting
	// ExecuteSqlRequest.query_mode.
	Stats *ResultSetStats `json:"stats,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ResultSet) MarshalJSON() ([]byte, error) {
	type noMethod ResultSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ResultSetMetadata: Metadata about a ResultSet or PartialResultSet.
type ResultSetMetadata struct {
	// RowType: Indicates the field names and types for the rows in the
	// result
	// set.  For example, a SQL query like "SELECT UserId, UserName
	// FROM
	// Users" could return a `row_type` value like:
	//
	//     "fields": [
	//       { "name": "UserId", "type": { "code": "INT64" } },
	//       { "name": "UserName", "type": { "code": "STRING" } },
	//     ]
	RowType *StructType `json:"rowType,omitempty"`

	// Transaction: If the read or SQL query began a transaction as a
	// side-effect, the
	// information about the new transaction is yielded here.
	Transaction *Transaction `json:"transaction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "RowType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "RowType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ResultSetMetadata) MarshalJSON() ([]byte, error) {
	type noMethod ResultSetMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ResultSetStats: Additional statistics about a ResultSet or
// PartialResultSet.
type ResultSetStats struct {
	// QueryPlan: QueryPlan for the query associated with this result.
	QueryPlan *QueryPlan `json:"queryPlan,omitempty"`

	// QueryStats: Aggregated statistics from the execution of the query.
	// Only present when
	// the query is profiled. For example, a query could return the
	// statistics as
	// follows:
	//
	//     {
	//       "rows_returned": "3",
	//       "elapsed_time": "1.22 secs",
	//       "cpu_time": "1.19 secs"
	//     }
	QueryStats googleapi.RawMessage `json:"queryStats,omitempty"`

	// ForceSendFields is a list of field names (e.g. "QueryPlan") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "QueryPlan") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ResultSetStats) MarshalJSON() ([]byte, error) {
	type noMethod ResultSetStats
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RollbackRequest: The request for Rollback.
type RollbackRequest struct {
	// TransactionId: Required. The transaction to roll back.
	TransactionId string `json:"transactionId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TransactionId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TransactionId") to include
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

// Session: A session in the Cloud Spanner API.
type Session struct {
	// Name: Required. The name of the session.
	Name string `json:"name,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *Session) MarshalJSON() ([]byte, error) {
	type noMethod Session
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SetIamPolicyRequest: Request message for `SetIamPolicy` method.
type SetIamPolicyRequest struct {
	// Policy: REQUIRED: The complete policy to be applied to the
	// `resource`. The size of
	// the policy is limited to a few 10s of KB. An empty policy is a
	// valid policy but certain Cloud Platform services (such as
	// Projects)
	// might reject them.
	Policy *Policy `json:"policy,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Policy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Policy") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SetIamPolicyRequest) MarshalJSON() ([]byte, error) {
	type noMethod SetIamPolicyRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ShortRepresentation: Condensed representation of a node and its
// subtree. Only present for
// `SCALAR` PlanNode(s).
type ShortRepresentation struct {
	// Description: A string representation of the expression subtree rooted
	// at this node.
	Description string `json:"description,omitempty"`

	// Subqueries: A mapping of (subquery variable name) -> (subquery node
	// id) for cases
	// where the `description` string of this node references a
	// `SCALAR`
	// subquery contained in the expression subtree rooted at this node.
	// The
	// referenced `SCALAR` subquery may not necessarily be a direct child
	// of
	// this node.
	Subqueries map[string]int64 `json:"subqueries,omitempty"`

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

func (s *ShortRepresentation) MarshalJSON() ([]byte, error) {
	type noMethod ShortRepresentation
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

// StructType: `StructType` defines the fields of a STRUCT type.
type StructType struct {
	// Fields: The list of fields that make up this struct. Order
	// is
	// significant, because values of this struct type are represented
	// as
	// lists, where the order of field values matches the order of
	// fields in the StructType. In turn, the order of fields
	// matches the order of columns in a read request, or the order
	// of
	// fields in the `SELECT` clause of a query.
	Fields []*Field `json:"fields,omitempty"`

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

func (s *StructType) MarshalJSON() ([]byte, error) {
	type noMethod StructType
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestIamPermissionsRequest: Request message for `TestIamPermissions`
// method.
type TestIamPermissionsRequest struct {
	// Permissions: REQUIRED: The set of permissions to check for
	// 'resource'.
	// Permissions with wildcards (such as '*', 'spanner.*',
	// 'spanner.instances.*') are not allowed.
	Permissions []string `json:"permissions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Permissions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestIamPermissionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestIamPermissionsResponse: Response message for `TestIamPermissions`
// method.
type TestIamPermissionsResponse struct {
	// Permissions: A subset of `TestPermissionsRequest.permissions` that
	// the caller is
	// allowed.
	Permissions []string `json:"permissions,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Permissions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestIamPermissionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Transaction: A transaction.
type Transaction struct {
	// Id: `id` may be used to identify the transaction in
	// subsequent
	// Read,
	// ExecuteSql,
	// Commit, or
	// Rollback calls.
	//
	// Single-use read-only transactions do not have IDs, because
	// single-use transactions do not support multiple requests.
	Id string `json:"id,omitempty"`

	// ReadTimestamp: For snapshot read-only transactions, the read
	// timestamp chosen
	// for the transaction. Not returned by default:
	// see
	// TransactionOptions.ReadOnly.return_read_timestamp.
	ReadTimestamp string `json:"readTimestamp,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *Transaction) MarshalJSON() ([]byte, error) {
	type noMethod Transaction
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TransactionOptions: # Transactions
//
//
// Each session can have at most one active transaction at a time. After
// the
// active transaction is completed, the session can immediately
// be
// re-used for the next transaction. It is not necessary to create a
// new session for each transaction.
//
// # Transaction Modes
//
// Cloud Spanner supports two transaction modes:
//
//   1. Locking read-write. This type of transaction is the only way
//      to write data into Cloud Spanner. These transactions rely on
//      pessimistic locking and, if necessary, two-phase commit.
//      Locking read-write transactions may abort, requiring the
//      application to retry.
//
//   2. Snapshot read-only. This transaction type provides guaranteed
//      consistency across several reads, but does not allow
//      writes. Snapshot read-only transactions can be configured to
//      read at timestamps in the past. Snapshot read-only
//      transactions do not need to be committed.
//
// For transactions that only read, snapshot read-only
// transactions
// provide simpler semantics and are almost always faster.
// In
// particular, read-only transactions do not take locks, so they do
// not conflict with read-write transactions. As a consequence of
// not
// taking locks, they also do not abort, so retry loops are not
// needed.
//
// Transactions may only read/write data in a single database. They
// may, however, read/write data in different tables within
// that
// database.
//
// ## Locking Read-Write Transactions
//
// Locking transactions may be used to atomically read-modify-write
// data anywhere in a database. This type of transaction is
// externally
// consistent.
//
// Clients should attempt to minimize the amount of time a
// transaction
// is active. Faster transactions commit with higher probability
// and cause less contention. Cloud Spanner attempts to keep read
// locks
// active as long as the transaction continues to do reads, and
// the
// transaction has not been terminated by
// Commit or
// Rollback.  Long periods of
// inactivity at the client may cause Cloud Spanner to release
// a
// transaction's locks and abort it.
//
// Reads performed within a transaction acquire locks on the data
// being read. Writes can only be done at commit time, after all
// reads
// have been completed.
// Conceptually, a read-write transaction consists of zero or more
// reads or SQL queries followed by
// Commit. At any time before
// Commit, the client can send a
// Rollback request to abort the
// transaction.
//
// ### Semantics
//
// Cloud Spanner can commit the transaction if all read locks it
// acquired
// are still valid at commit time, and it is able to acquire write
// locks for all writes. Cloud Spanner can abort the transaction for
// any
// reason. If a commit attempt returns `ABORTED`, Cloud Spanner
// guarantees
// that the transaction has not modified any user data in Cloud
// Spanner.
//
// Unless the transaction commits, Cloud Spanner makes no guarantees
// about
// how long the transaction's locks were held for. It is an error to
// use Cloud Spanner locks for any sort of mutual exclusion other
// than
// between Cloud Spanner transactions themselves.
//
// ### Retrying Aborted Transactions
//
// When a transaction aborts, the application can choose to retry
// the
// whole transaction again. To maximize the chances of
// successfully
// committing the retry, the client should execute the retry in the
// same session as the original attempt. The original session's
// lock
// priority increases with each consecutive abort, meaning that
// each
// attempt has a slightly better chance of success than the
// previous.
//
// Under some circumstances (e.g., many transactions attempting
// to
// modify the same row(s)), a transaction can abort many times in
// a
// short period before successfully committing. Thus, it is not a
// good
// idea to cap the number of retries a transaction can attempt;
// instead, it is better to limit the total amount of wall time
// spent
// retrying.
//
// ### Idle Transactions
//
// A transaction is considered idle if it has no outstanding reads
// or
// SQL queries and has not started a read or SQL query within the last
// 10
// seconds. Idle transactions can be aborted by Cloud Spanner so that
// they
// don't hold on to locks indefinitely. In that case, the commit
// will
// fail with error `ABORTED`.
//
// If this behavior is undesirable, periodically executing a simple
// SQL query in the transaction (e.g., `SELECT 1`) prevents
// the
// transaction from becoming idle.
//
// ## Snapshot Read-Only Transactions
//
// Snapshot read-only transactions provides a simpler method
// than
// locking read-write transactions for doing several consistent
// reads. However, this type of transaction does not support
// writes.
//
// Snapshot transactions do not take locks. Instead, they work
// by
// choosing a Cloud Spanner timestamp, then executing all reads at
// that
// timestamp. Since they do not acquire locks, they do not
// block
// concurrent read-write transactions.
//
// Unlike locking read-write transactions, snapshot
// read-only
// transactions never abort. They can fail if the chosen read
// timestamp is garbage collected; however, the default
// garbage
// collection policy is generous enough that most applications do
// not
// need to worry about this in practice.
//
// Snapshot read-only transactions do not need to call
// Commit or
// Rollback (and in fact are not
// permitted to do so).
//
// To execute a snapshot transaction, the client specifies a
// timestamp
// bound, which tells Cloud Spanner how to choose a read timestamp.
//
// The types of timestamp bound are:
//
//   - Strong (the default).
//   - Bounded staleness.
//   - Exact staleness.
//
// If the Cloud Spanner database to be read is geographically
// distributed,
// stale read-only transactions can execute more quickly than strong
// or read-write transaction, because they are able to execute far
// from the leader replica.
//
// Each type of timestamp bound is discussed in detail below.
//
// ### Strong
//
// Strong reads are guaranteed to see the effects of all
// transactions
// that have committed before the start of the read. Furthermore,
// all
// rows yielded by a single read are consistent with each other --
// if
// any part of the read observes a transaction, all parts of the
// read
// see the transaction.
//
// Strong reads are not repeatable: two consecutive strong
// read-only
// transactions might return inconsistent results if there
// are
// concurrent writes. If consistency across reads is required, the
// reads should be executed within a transaction or at an exact
// read
// timestamp.
//
// See TransactionOptions.ReadOnly.strong.
//
// ### Exact Staleness
//
// These timestamp bounds execute reads at a user-specified
// timestamp. Reads at a timestamp are guaranteed to see a
// consistent
// prefix of the global transaction history: they observe
// modifications done by all transactions with a commit timestamp <=
// the read timestamp, and observe none of the modifications done
// by
// transactions with a larger commit timestamp. They will block
// until
// all conflicting transactions that may be assigned commit
// timestamps
// <= the read timestamp have finished.
//
// The timestamp can either be expressed as an absolute Cloud Spanner
// commit
// timestamp or a staleness relative to the current time.
//
// These modes do not require a "negotiation phase" to pick a
// timestamp. As a result, they execute slightly faster than
// the
// equivalent boundedly stale concurrency modes. On the other
// hand,
// boundedly stale reads usually return fresher results.
//
// See TransactionOptions.ReadOnly.read_timestamp
// and
// TransactionOptions.ReadOnly.exact_staleness.
//
// ### Bounded Staleness
//
// Bounded staleness modes allow Cloud Spanner to pick the read
// timestamp,
// subject to a user-provided staleness bound. Cloud Spanner chooses
// the
// newest timestamp within the staleness bound that allows execution
// of the reads at the closest available replica without blocking.
//
// All rows yielded are consistent with each other -- if any part of
// the read observes a transaction, all parts of the read see
// the
// transaction. Boundedly stale reads are not repeatable: two
// stale
// reads, even if they use the same staleness bound, can execute
// at
// different timestamps and thus return inconsistent results.
//
// Boundedly stale reads execute in two phases: the first
// phase
// negotiates a timestamp among all replicas needed to serve the
// read. In the second phase, reads are executed at the
// negotiated
// timestamp.
//
// As a result of the two phase execution, bounded staleness reads
// are
// usually a little slower than comparable exact staleness
// reads. However, they are typically able to return fresher
// results, and are more likely to execute at the closest
// replica.
//
// Because the timestamp negotiation requires up-front knowledge
// of
// which rows will be read, it can only be used with
// single-use
// read-only transactions.
//
// See TransactionOptions.ReadOnly.max_staleness
// and
// TransactionOptions.ReadOnly.min_read_timestamp.
//
// ### Old Read Timestamps and Garbage Collection
//
// Cloud Spanner continuously garbage collects deleted and overwritten
// data
// in the background to reclaim storage space. This process is known
// as "version GC". By default, version GC reclaims versions after
// they
// are one hour old. Because of this, Cloud Spanner cannot perform
// reads
// at read timestamps more than one hour in the past. This
// restriction also applies to in-progress reads and/or SQL queries
// whose
// timestamp become too old while executing. Reads and SQL queries
// with
// too-old read timestamps fail with the error `FAILED_PRECONDITION`.
type TransactionOptions struct {
	// ReadOnly: Transaction will not write.
	//
	// Authorization to begin a read-only transaction
	// requires
	// `spanner.databases.beginReadOnlyTransaction` permission
	// on the `session` resource.
	ReadOnly *ReadOnly `json:"readOnly,omitempty"`

	// ReadWrite: Transaction may write.
	//
	// Authorization to begin a read-write transaction
	// requires
	// `spanner.databases.beginOrRollbackReadWriteTransaction` permission
	// on the `session` resource.
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

// TransactionSelector: This message is used to select the transaction
// in which a
// Read or
// ExecuteSql call runs.
//
// See TransactionOptions for more information about transactions.
type TransactionSelector struct {
	// Begin: Begin a new transaction and execute this read or SQL query
	// in
	// it. The transaction ID of the new transaction is returned
	// in
	// ResultSetMetadata.transaction, which is a Transaction.
	Begin *TransactionOptions `json:"begin,omitempty"`

	// Id: Execute the read or SQL query in a previously-started
	// transaction.
	Id string `json:"id,omitempty"`

	// SingleUse: Execute the read or SQL query in a temporary
	// transaction.
	// This is the most efficient way to execute a transaction that
	// consists of a single SQL query.
	SingleUse *TransactionOptions `json:"singleUse,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Begin") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Begin") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TransactionSelector) MarshalJSON() ([]byte, error) {
	type noMethod TransactionSelector
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Type: `Type` indicates the type of a Cloud Spanner value, as might be
// stored in a
// table cell or returned from an SQL query.
type Type struct {
	// ArrayElementType: If code == ARRAY, then `array_element_type`
	// is the type of the array elements.
	ArrayElementType *Type `json:"arrayElementType,omitempty"`

	// Code: Required. The TypeCode for this type.
	//
	// Possible values:
	//   "TYPE_CODE_UNSPECIFIED" - Not specified.
	//   "BOOL" - Encoded as JSON `true` or `false`.
	//   "INT64" - Encoded as `string`, in decimal format.
	//   "FLOAT64" - Encoded as `number`, or the strings "NaN",
	// "Infinity", or
	// "-Infinity".
	//   "TIMESTAMP" - Encoded as `string` in RFC 3339 timestamp format. The
	// time zone
	// must be present, and must be "Z".
	//   "DATE" - Encoded as `string` in RFC 3339 date format.
	//   "STRING" - Encoded as `string`.
	//   "BYTES" - Encoded as a base64-encoded `string`, as described in RFC
	// 4648,
	// section 4.
	//   "ARRAY" - Encoded as `list`, where the list elements are
	// represented
	// according to array_element_type.
	//   "STRUCT" - Encoded as `list`, where list element `i` is represented
	// according
	// to [struct_type.fields[i]][google.spanner.v1.StructType.fields].
	Code string `json:"code,omitempty"`

	// StructType: If code == STRUCT, then `struct_type`
	// provides type information for the struct's fields.
	StructType *StructType `json:"structType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArrayElementType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ArrayElementType") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Type) MarshalJSON() ([]byte, error) {
	type noMethod Type
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateDatabaseDdlMetadata: Metadata type for the operation returned
// by
// UpdateDatabaseDdl.
type UpdateDatabaseDdlMetadata struct {
	// CommitTimestamps: Reports the commit timestamps of all statements
	// that have
	// succeeded so far, where `commit_timestamps[i]` is the
	// commit
	// timestamp for the statement `statements[i]`.
	CommitTimestamps []string `json:"commitTimestamps,omitempty"`

	// Database: The database being modified.
	Database string `json:"database,omitempty"`

	// Statements: For an update this list contains all the statements. For
	// an
	// individual statement, this list contains only that statement.
	Statements []string `json:"statements,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommitTimestamps") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CommitTimestamps") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *UpdateDatabaseDdlMetadata) MarshalJSON() ([]byte, error) {
	type noMethod UpdateDatabaseDdlMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateDatabaseDdlRequest: Enqueues the given DDL statements to be
// applied, in order but not
// necessarily all at once, to the database schema at some point
// (or
// points) in the future. The server checks that the statements
// are executable (syntactically valid, name tables that exist,
// etc.)
// before enqueueing them, but they may still fail upon
// later execution (e.g., if a statement from another batch
// of
// statements is applied first and it conflicts in some way, or if
// there is some data-related problem like a `NULL` value in a column
// to
// which `NOT NULL` would be added). If a statement fails,
// all
// subsequent statements in the batch are automatically cancelled.
//
// Each batch of statements is assigned a name which can be used
// with
// the Operations API to monitor
// progress. See the
// operation_id field for more
// details.
type UpdateDatabaseDdlRequest struct {
	// OperationId: If empty, the new update request is assigned
	// an
	// automatically-generated operation ID. Otherwise, `operation_id`
	// is used to construct the name of the resulting
	// Operation.
	//
	// Specifying an explicit operation ID simplifies determining
	// whether the statements were executed in the event that
	// the
	// UpdateDatabaseDdl call is replayed,
	// or the return value is otherwise lost: the database
	// and
	// `operation_id` fields can be combined to form the
	// name of the resulting
	// longrunning.Operation:
	// `<database>/operations/<operation_id>`.
	//
	// `operation_id` should be unique within the database, and must be
	// a valid identifier: `a-z*`. Note that
	// automatically-generated operation IDs always begin with
	// an
	// underscore. If the named operation already exists,
	// UpdateDatabaseDdl returns
	// `ALREADY_EXISTS`.
	OperationId string `json:"operationId,omitempty"`

	// Statements: DDL statements to be applied to the database.
	Statements []string `json:"statements,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateDatabaseDdlRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateDatabaseDdlRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateInstanceMetadata: Metadata type for the operation returned
// by
// UpdateInstance.
type UpdateInstanceMetadata struct {
	// CancelTime: The time at which this operation was cancelled. If set,
	// this operation is
	// in the process of undoing itself (which is guaranteed to succeed)
	// and
	// cannot be cancelled again.
	CancelTime string `json:"cancelTime,omitempty"`

	// EndTime: The time at which this operation failed or was completed
	// successfully.
	EndTime string `json:"endTime,omitempty"`

	// Instance: The desired end state of the update.
	Instance *Instance `json:"instance,omitempty"`

	// StartTime: The time at which UpdateInstance
	// request was received.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CancelTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CancelTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateInstanceMetadata) MarshalJSON() ([]byte, error) {
	type noMethod UpdateInstanceMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateInstanceRequest: The request for UpdateInstance.
type UpdateInstanceRequest struct {
	// FieldMask: Required. A mask specifying which fields in
	// [][google.spanner.admin.instance.v1.UpdateInstanceRequest.instance]
	// should be updated.
	// The field mask must always be specified; this prevents any future
	// fields in
	// [][google.spanner.admin.instance.v1.Instance] from being erased
	// accidentally by clients that do not know
	// about them.
	FieldMask string `json:"fieldMask,omitempty"`

	// Instance: Required. The instance to update, which must always include
	// the instance
	// name.  Otherwise, only fields mentioned in
	// [][google.spanner.admin.instance.v1.UpdateInstanceRequest.field_mask]
	// need be included.
	Instance *Instance `json:"instance,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FieldMask") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FieldMask") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateInstanceRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateInstanceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Write: Arguments to insert, update, insert_or_update, and
// replace operations.
type Write struct {
	// Columns: The names of the columns in table to be written.
	//
	// The list of columns must contain enough columns to allow
	// Cloud Spanner to derive values for all primary key columns in
	// the
	// row(s) to be modified.
	Columns []string `json:"columns,omitempty"`

	// Table: Required. The table whose rows will be written.
	Table string `json:"table,omitempty"`

	// Values: The values to be written. `values` can contain more than
	// one
	// list of values. If it does, then multiple rows are written, one
	// for each entry in `values`. Each list in `values` must have
	// exactly as many entries as there are entries in columns
	// above. Sending multiple lists is equivalent to sending
	// multiple
	// `Mutation`s, each containing one `values` entry and repeating
	// table and columns. Individual values in each list are
	// encoded as described here.
	Values [][]interface{} `json:"values,omitempty"`

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

func (s *Write) MarshalJSON() ([]byte, error) {
	type noMethod Write
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "spanner.projects.instanceConfigs.get":

type ProjectsInstanceConfigsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets information about a particular instance configuration.
func (r *ProjectsInstanceConfigsService) Get(name string) *ProjectsInstanceConfigsGetCall {
	c := &ProjectsInstanceConfigsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstanceConfigsGetCall) Fields(s ...googleapi.Field) *ProjectsInstanceConfigsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstanceConfigsGetCall) IfNoneMatch(entityTag string) *ProjectsInstanceConfigsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstanceConfigsGetCall) Context(ctx context.Context) *ProjectsInstanceConfigsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstanceConfigsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstanceConfigsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instanceConfigs.get" call.
// Exactly one of *InstanceConfig or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *InstanceConfig.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstanceConfigsGetCall) Do(opts ...googleapi.CallOption) (*InstanceConfig, error) {
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
	ret := &InstanceConfig{
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
	//   "description": "Gets information about a particular instance configuration.",
	//   "flatPath": "v1/projects/{projectsId}/instanceConfigs/{instanceConfigsId}",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instanceConfigs.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The name of the requested instance configuration. Values are of\nthe form `projects/\u003cproject\u003e/instanceConfigs/\u003cconfig\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instanceConfigs/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "InstanceConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instanceConfigs.list":

type ProjectsInstanceConfigsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the supported instance configurations for a given
// project.
func (r *ProjectsInstanceConfigsService) List(parent string) *ProjectsInstanceConfigsListCall {
	c := &ProjectsInstanceConfigsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": Number of instance
// configurations to be returned in the response. If 0 or
// less, defaults to the server's maximum allowed page size.
func (c *ProjectsInstanceConfigsListCall) PageSize(pageSize int64) *ProjectsInstanceConfigsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If non-empty,
// `page_token` should contain a
// next_page_token
// from a previous ListInstanceConfigsResponse.
func (c *ProjectsInstanceConfigsListCall) PageToken(pageToken string) *ProjectsInstanceConfigsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstanceConfigsListCall) Fields(s ...googleapi.Field) *ProjectsInstanceConfigsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstanceConfigsListCall) IfNoneMatch(entityTag string) *ProjectsInstanceConfigsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstanceConfigsListCall) Context(ctx context.Context) *ProjectsInstanceConfigsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstanceConfigsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstanceConfigsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/instanceConfigs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instanceConfigs.list" call.
// Exactly one of *ListInstanceConfigsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListInstanceConfigsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstanceConfigsListCall) Do(opts ...googleapi.CallOption) (*ListInstanceConfigsResponse, error) {
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
	ret := &ListInstanceConfigsResponse{
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
	//   "description": "Lists the supported instance configurations for a given project.",
	//   "flatPath": "v1/projects/{projectsId}/instanceConfigs",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instanceConfigs.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Number of instance configurations to be returned in the response. If 0 or\nless, defaults to the server's maximum allowed page size.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "If non-empty, `page_token` should contain a\nnext_page_token\nfrom a previous ListInstanceConfigsResponse.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The name of the project for which a list of supported instance\nconfigurations is requested. Values are of the form\n`projects/\u003cproject\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/instanceConfigs",
	//   "response": {
	//     "$ref": "ListInstanceConfigsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsInstanceConfigsListCall) Pages(ctx context.Context, f func(*ListInstanceConfigsResponse) error) error {
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

// method id "spanner.projects.instances.create":

type ProjectsInstancesCreateCall struct {
	s                     *Service
	parent                string
	createinstancerequest *CreateInstanceRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Create: Creates an instance and begins preparing it to begin serving.
// The
// returned long-running operation
// can be used to track the progress of preparing the new
// instance. The instance name is assigned by the caller. If the
// named instance already exists, `CreateInstance`
// returns
// `ALREADY_EXISTS`.
//
// Immediately upon completion of this request:
//
//   * The instance is readable via the API, with all requested
// attributes
//     but no allocated resources. Its state is `CREATING`.
//
// Until completion of the returned operation:
//
//   * Cancelling the operation renders the instance immediately
// unreadable
//     via the API.
//   * The instance can be deleted.
//   * All other attempts to modify the instance are rejected.
//
// Upon completion of the returned operation:
//
//   * Billing for all successfully-allocated resources begins (some
// types
//     may have lower than the requested levels).
//   * Databases can be created in the instance.
//   * The instance's allocated resource levels are readable via the
// API.
//   * The instance's state becomes `READY`.
//
// The returned long-running operation will
// have a name of the format `<instance_name>/operations/<operation_id>`
// and
// can be used to track creation of the instance.  The
// metadata field type is
// CreateInstanceMetadata.
// The response field type is
// Instance, if successful.
func (r *ProjectsInstancesService) Create(parent string, createinstancerequest *CreateInstanceRequest) *ProjectsInstancesCreateCall {
	c := &ProjectsInstancesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.createinstancerequest = createinstancerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesCreateCall) Fields(s ...googleapi.Field) *ProjectsInstancesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesCreateCall) Context(ctx context.Context) *ProjectsInstancesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createinstancerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/instances")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Creates an instance and begins preparing it to begin serving. The\nreturned long-running operation\ncan be used to track the progress of preparing the new\ninstance. The instance name is assigned by the caller. If the\nnamed instance already exists, `CreateInstance` returns\n`ALREADY_EXISTS`.\n\nImmediately upon completion of this request:\n\n  * The instance is readable via the API, with all requested attributes\n    but no allocated resources. Its state is `CREATING`.\n\nUntil completion of the returned operation:\n\n  * Cancelling the operation renders the instance immediately unreadable\n    via the API.\n  * The instance can be deleted.\n  * All other attempts to modify the instance are rejected.\n\nUpon completion of the returned operation:\n\n  * Billing for all successfully-allocated resources begins (some types\n    may have lower than the requested levels).\n  * Databases can be created in the instance.\n  * The instance's allocated resource levels are readable via the API.\n  * The instance's state becomes `READY`.\n\nThe returned long-running operation will\nhave a name of the format `\u003cinstance_name\u003e/operations/\u003coperation_id\u003e` and\ncan be used to track creation of the instance.  The\nmetadata field type is\nCreateInstanceMetadata.\nThe response field type is\nInstance, if successful.",
	//   "flatPath": "v1/projects/{projectsId}/instances",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "Required. The name of the project in which to create the instance. Values\nare of the form `projects/\u003cproject\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/instances",
	//   "request": {
	//     "$ref": "CreateInstanceRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.delete":

type ProjectsInstancesDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes an instance.
//
// Immediately upon completion of the request:
//
//   * Billing ceases for all of the instance's reserved
// resources.
//
// Soon afterward:
//
//   * The instance and *all of its databases* immediately and
//     irrevocably disappear from the API. All data in the databases
//     is permanently deleted.
func (r *ProjectsInstancesService) Delete(name string) *ProjectsInstancesDeleteCall {
	c := &ProjectsInstancesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDeleteCall) Fields(s ...googleapi.Field) *ProjectsInstancesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDeleteCall) Context(ctx context.Context) *ProjectsInstancesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes an instance.\n\nImmediately upon completion of the request:\n\n  * Billing ceases for all of the instance's reserved resources.\n\nSoon afterward:\n\n  * The instance and *all of its databases* immediately and\n    irrevocably disappear from the API. All data in the databases\n    is permanently deleted.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}",
	//   "httpMethod": "DELETE",
	//   "id": "spanner.projects.instances.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The name of the instance to be deleted. Values are of the form\n`projects/\u003cproject\u003e/instances/\u003cinstance\u003e`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
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
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.get":

type ProjectsInstancesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets information about a particular instance.
func (r *ProjectsInstancesService) Get(name string) *ProjectsInstancesGetCall {
	c := &ProjectsInstancesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesGetCall) Fields(s ...googleapi.Field) *ProjectsInstancesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesGetCall) IfNoneMatch(entityTag string) *ProjectsInstancesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesGetCall) Context(ctx context.Context) *ProjectsInstancesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.get" call.
// Exactly one of *Instance or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Instance.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesGetCall) Do(opts ...googleapi.CallOption) (*Instance, error) {
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
	ret := &Instance{
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
	//   "description": "Gets information about a particular instance.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The name of the requested instance. Values are of the form\n`projects/\u003cproject\u003e/instances/\u003cinstance\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Instance"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.getIamPolicy":

type ProjectsInstancesGetIamPolicyCall struct {
	s                   *Service
	resource            string
	getiampolicyrequest *GetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// GetIamPolicy: Gets the access control policy for an instance
// resource. Returns an empty
// policy if an instance exists but does not have a policy
// set.
//
// Authorization requires `spanner.instances.getIamPolicy` on
// resource.
func (r *ProjectsInstancesService) GetIamPolicy(resource string, getiampolicyrequest *GetIamPolicyRequest) *ProjectsInstancesGetIamPolicyCall {
	c := &ProjectsInstancesGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.getiampolicyrequest = getiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesGetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsInstancesGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesGetIamPolicyCall) Context(ctx context.Context) *ProjectsInstancesGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.getiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Gets the access control policy for an instance resource. Returns an empty\npolicy if an instance exists but does not have a policy set.\n\nAuthorization requires `spanner.instances.getIamPolicy` on\nresource.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The Cloud Spanner resource for which the policy is being retrieved. The format is `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e` for instance resources and `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e/databases/\u003cdatabase ID\u003e` for database resources.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:getIamPolicy",
	//   "request": {
	//     "$ref": "GetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.list":

type ProjectsInstancesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all instances in the given project.
func (r *ProjectsInstancesService) List(parent string) *ProjectsInstancesListCall {
	c := &ProjectsInstancesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// Filter sets the optional parameter "filter": An expression for
// filtering the results of the request. Filter rules are
// case insensitive. The fields eligible for filtering are:
//
//   * name
//   * display_name
//   * labels.key where key is the name of a label
//
// Some examples of using filters are:
//
//   * name:* --> The instance has a name.
//   * name:Howl --> The instance's name contains the string "howl".
//   * name:HOWL --> Equivalent to above.
//   * NAME:howl --> Equivalent to above.
//   * labels.env:* --> The instance has the label "env".
//   * labels.env:dev --> The instance has the label "env" and the value
// of
//                        the label contains the string "dev".
//   * name:howl labels.env:dev --> The instance's name contains "howl"
// and
//                                  it has the label "env" with its
// value
//                                  containing "dev".
func (c *ProjectsInstancesListCall) Filter(filter string) *ProjectsInstancesListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": Number of instances
// to be returned in the response. If 0 or less, defaults
// to the server's maximum allowed page size.
func (c *ProjectsInstancesListCall) PageSize(pageSize int64) *ProjectsInstancesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If non-empty,
// `page_token` should contain a
// next_page_token from a
// previous ListInstancesResponse.
func (c *ProjectsInstancesListCall) PageToken(pageToken string) *ProjectsInstancesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesListCall) Fields(s ...googleapi.Field) *ProjectsInstancesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesListCall) IfNoneMatch(entityTag string) *ProjectsInstancesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesListCall) Context(ctx context.Context) *ProjectsInstancesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/instances")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.list" call.
// Exactly one of *ListInstancesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListInstancesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesListCall) Do(opts ...googleapi.CallOption) (*ListInstancesResponse, error) {
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
	ret := &ListInstancesResponse{
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
	//   "description": "Lists all instances in the given project.",
	//   "flatPath": "v1/projects/{projectsId}/instances",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "An expression for filtering the results of the request. Filter rules are\ncase insensitive. The fields eligible for filtering are:\n\n  * name\n  * display_name\n  * labels.key where key is the name of a label\n\nSome examples of using filters are:\n\n  * name:* --\u003e The instance has a name.\n  * name:Howl --\u003e The instance's name contains the string \"howl\".\n  * name:HOWL --\u003e Equivalent to above.\n  * NAME:howl --\u003e Equivalent to above.\n  * labels.env:* --\u003e The instance has the label \"env\".\n  * labels.env:dev --\u003e The instance has the label \"env\" and the value of\n                       the label contains the string \"dev\".\n  * name:howl labels.env:dev --\u003e The instance's name contains \"howl\" and\n                                 it has the label \"env\" with its value\n                                 containing \"dev\".",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Number of instances to be returned in the response. If 0 or less, defaults\nto the server's maximum allowed page size.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "If non-empty, `page_token` should contain a\nnext_page_token from a\nprevious ListInstancesResponse.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The name of the project for which a list of instances is\nrequested. Values are of the form `projects/\u003cproject\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/instances",
	//   "response": {
	//     "$ref": "ListInstancesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsInstancesListCall) Pages(ctx context.Context, f func(*ListInstancesResponse) error) error {
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

// method id "spanner.projects.instances.patch":

type ProjectsInstancesPatchCall struct {
	s                     *Service
	nameid                string
	updateinstancerequest *UpdateInstanceRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Patch: Updates an instance, and begins allocating or releasing
// resources
// as requested. The returned long-running
// operation can be used to track the
// progress of updating the instance. If the named instance does
// not
// exist, returns `NOT_FOUND`.
//
// Immediately upon completion of this request:
//
//   * For resource types for which a decrease in the instance's
// allocation
//     has been requested, billing is based on the newly-requested
// level.
//
// Until completion of the returned operation:
//
//   * Cancelling the operation sets its metadata's
//     cancel_time, and begins
//     restoring resources to their pre-request values. The operation
//     is guaranteed to succeed at undoing all resource changes,
//     after which point it terminates with a `CANCELLED` status.
//   * All other attempts to modify the instance are rejected.
//   * Reading the instance via the API continues to give the
// pre-request
//     resource levels.
//
// Upon completion of the returned operation:
//
//   * Billing begins for all successfully-allocated resources (some
// types
//     may have lower than the requested levels).
//   * All newly-reserved resources are available for serving the
// instance's
//     tables.
//   * The instance's new resource levels are readable via the API.
//
// The returned long-running operation will
// have a name of the format `<instance_name>/operations/<operation_id>`
// and
// can be used to track the instance modification.  The
// metadata field type is
// UpdateInstanceMetadata.
// The response field type is
// Instance, if successful.
//
// Authorization requires `spanner.instances.update` permission
// on
// resource name.
func (r *ProjectsInstancesService) Patch(nameid string, updateinstancerequest *UpdateInstanceRequest) *ProjectsInstancesPatchCall {
	c := &ProjectsInstancesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.nameid = nameid
	c.updateinstancerequest = updateinstancerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesPatchCall) Fields(s ...googleapi.Field) *ProjectsInstancesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesPatchCall) Context(ctx context.Context) *ProjectsInstancesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updateinstancerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.nameid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.patch" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesPatchCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Updates an instance, and begins allocating or releasing resources\nas requested. The returned long-running\noperation can be used to track the\nprogress of updating the instance. If the named instance does not\nexist, returns `NOT_FOUND`.\n\nImmediately upon completion of this request:\n\n  * For resource types for which a decrease in the instance's allocation\n    has been requested, billing is based on the newly-requested level.\n\nUntil completion of the returned operation:\n\n  * Cancelling the operation sets its metadata's\n    cancel_time, and begins\n    restoring resources to their pre-request values. The operation\n    is guaranteed to succeed at undoing all resource changes,\n    after which point it terminates with a `CANCELLED` status.\n  * All other attempts to modify the instance are rejected.\n  * Reading the instance via the API continues to give the pre-request\n    resource levels.\n\nUpon completion of the returned operation:\n\n  * Billing begins for all successfully-allocated resources (some types\n    may have lower than the requested levels).\n  * All newly-reserved resources are available for serving the instance's\n    tables.\n  * The instance's new resource levels are readable via the API.\n\nThe returned long-running operation will\nhave a name of the format `\u003cinstance_name\u003e/operations/\u003coperation_id\u003e` and\ncan be used to track the instance modification.  The\nmetadata field type is\nUpdateInstanceMetadata.\nThe response field type is\nInstance, if successful.\n\nAuthorization requires `spanner.instances.update` permission on\nresource name.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}",
	//   "httpMethod": "PATCH",
	//   "id": "spanner.projects.instances.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. A unique identifier for the instance, which cannot be changed\nafter the instance is created. Values are of the form\n`projects/\u003cproject\u003e/instances/a-z*[a-z0-9]`. The final\nsegment of the name must be between 6 and 30 characters in length.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "UpdateInstanceRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.setIamPolicy":

type ProjectsInstancesSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy on an instance resource.
// Replaces any
// existing policy.
//
// Authorization requires `spanner.instances.setIamPolicy` on
// resource.
func (r *ProjectsInstancesService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ProjectsInstancesSetIamPolicyCall {
	c := &ProjectsInstancesSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesSetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsInstancesSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesSetIamPolicyCall) Context(ctx context.Context) *ProjectsInstancesSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.setiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Sets the access control policy on an instance resource. Replaces any\nexisting policy.\n\nAuthorization requires `spanner.instances.setIamPolicy` on\nresource.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The Cloud Spanner resource for which the policy is being set. The format is `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e` for instance resources and `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e/databases/\u003cdatabase ID\u003e` for databases resources.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:setIamPolicy",
	//   "request": {
	//     "$ref": "SetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.testIamPermissions":

type ProjectsInstancesTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that the caller has on the
// specified instance resource.
//
// Attempting this RPC on a non-existent Cloud Spanner instance resource
// will
// result in a NOT_FOUND error if the user has
// `spanner.instances.list`
// permission on the containing Google Cloud Project. Otherwise returns
// an
// empty set of permissions.
func (r *ProjectsInstancesService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ProjectsInstancesTestIamPermissionsCall {
	c := &ProjectsInstancesTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesTestIamPermissionsCall) Fields(s ...googleapi.Field) *ProjectsInstancesTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesTestIamPermissionsCall) Context(ctx context.Context) *ProjectsInstancesTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testiampermissionsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	ret := &TestIamPermissionsResponse{
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
	//   "description": "Returns permissions that the caller has on the specified instance resource.\n\nAttempting this RPC on a non-existent Cloud Spanner instance resource will\nresult in a NOT_FOUND error if the user has `spanner.instances.list`\npermission on the containing Google Cloud Project. Otherwise returns an\nempty set of permissions.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The Cloud Spanner resource for which permissions are being tested. The format is `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e` for instance resources and `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e/databases/\u003cdatabase ID\u003e` for database resources.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:testIamPermissions",
	//   "request": {
	//     "$ref": "TestIamPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestIamPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.create":

type ProjectsInstancesDatabasesCreateCall struct {
	s                     *Service
	parent                string
	createdatabaserequest *CreateDatabaseRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Create: Creates a new Cloud Spanner database and starts to prepare it
// for serving.
// The returned long-running operation will
// have a name of the format `<database_name>/operations/<operation_id>`
// and
// can be used to track preparation of the database. The
// metadata field type is
// CreateDatabaseMetadata. The
// response field type is
// Database, if successful.
func (r *ProjectsInstancesDatabasesService) Create(parent string, createdatabaserequest *CreateDatabaseRequest) *ProjectsInstancesDatabasesCreateCall {
	c := &ProjectsInstancesDatabasesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.createdatabaserequest = createdatabaserequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesCreateCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesCreateCall) Context(ctx context.Context) *ProjectsInstancesDatabasesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createdatabaserequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/databases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Creates a new Cloud Spanner database and starts to prepare it for serving.\nThe returned long-running operation will\nhave a name of the format `\u003cdatabase_name\u003e/operations/\u003coperation_id\u003e` and\ncan be used to track preparation of the database. The\nmetadata field type is\nCreateDatabaseMetadata. The\nresponse field type is\nDatabase, if successful.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "Required. The name of the instance that will serve the new database.\nValues are of the form `projects/\u003cproject\u003e/instances/\u003cinstance\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/databases",
	//   "request": {
	//     "$ref": "CreateDatabaseRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.dropDatabase":

type ProjectsInstancesDatabasesDropDatabaseCall struct {
	s          *Service
	database   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// DropDatabase: Drops (aka deletes) a Cloud Spanner database.
func (r *ProjectsInstancesDatabasesService) DropDatabase(database string) *ProjectsInstancesDatabasesDropDatabaseCall {
	c := &ProjectsInstancesDatabasesDropDatabaseCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.database = database
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesDropDatabaseCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesDropDatabaseCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesDropDatabaseCall) Context(ctx context.Context) *ProjectsInstancesDatabasesDropDatabaseCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesDropDatabaseCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesDropDatabaseCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+database}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.dropDatabase" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesDropDatabaseCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Drops (aka deletes) a Cloud Spanner database.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}",
	//   "httpMethod": "DELETE",
	//   "id": "spanner.projects.instances.databases.dropDatabase",
	//   "parameterOrder": [
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Required. The database to be dropped.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+database}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.get":

type ProjectsInstancesDatabasesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the state of a Cloud Spanner database.
func (r *ProjectsInstancesDatabasesService) Get(name string) *ProjectsInstancesDatabasesGetCall {
	c := &ProjectsInstancesDatabasesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesGetCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesDatabasesGetCall) IfNoneMatch(entityTag string) *ProjectsInstancesDatabasesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesGetCall) Context(ctx context.Context) *ProjectsInstancesDatabasesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.databases.get" call.
// Exactly one of *Database or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Database.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesGetCall) Do(opts ...googleapi.CallOption) (*Database, error) {
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
	ret := &Database{
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
	//   "description": "Gets the state of a Cloud Spanner database.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.databases.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The name of the requested database. Values are of the form\n`projects/\u003cproject\u003e/instances/\u003cinstance\u003e/databases/\u003cdatabase\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Database"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.getDdl":

type ProjectsInstancesDatabasesGetDdlCall struct {
	s            *Service
	database     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetDdl: Returns the schema of a Cloud Spanner database as a list of
// formatted
// DDL statements. This method does not show pending schema updates,
// those may
// be queried using the Operations API.
func (r *ProjectsInstancesDatabasesService) GetDdl(database string) *ProjectsInstancesDatabasesGetDdlCall {
	c := &ProjectsInstancesDatabasesGetDdlCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.database = database
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesGetDdlCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesGetDdlCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesDatabasesGetDdlCall) IfNoneMatch(entityTag string) *ProjectsInstancesDatabasesGetDdlCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesGetDdlCall) Context(ctx context.Context) *ProjectsInstancesDatabasesGetDdlCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesGetDdlCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesGetDdlCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+database}/ddl")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.getDdl" call.
// Exactly one of *GetDatabaseDdlResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetDatabaseDdlResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesGetDdlCall) Do(opts ...googleapi.CallOption) (*GetDatabaseDdlResponse, error) {
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
	ret := &GetDatabaseDdlResponse{
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
	//   "description": "Returns the schema of a Cloud Spanner database as a list of formatted\nDDL statements. This method does not show pending schema updates, those may\nbe queried using the Operations API.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/ddl",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.databases.getDdl",
	//   "parameterOrder": [
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Required. The database whose schema we wish to get.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+database}/ddl",
	//   "response": {
	//     "$ref": "GetDatabaseDdlResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.getIamPolicy":

type ProjectsInstancesDatabasesGetIamPolicyCall struct {
	s                   *Service
	resource            string
	getiampolicyrequest *GetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// GetIamPolicy: Gets the access control policy for a database resource.
// Returns an empty
// policy if a database exists but does not have a policy
// set.
//
// Authorization requires `spanner.databases.getIamPolicy` permission
// on
// resource.
func (r *ProjectsInstancesDatabasesService) GetIamPolicy(resource string, getiampolicyrequest *GetIamPolicyRequest) *ProjectsInstancesDatabasesGetIamPolicyCall {
	c := &ProjectsInstancesDatabasesGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.getiampolicyrequest = getiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesGetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesGetIamPolicyCall) Context(ctx context.Context) *ProjectsInstancesDatabasesGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.getiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Gets the access control policy for a database resource. Returns an empty\npolicy if a database exists but does not have a policy set.\n\nAuthorization requires `spanner.databases.getIamPolicy` permission on\nresource.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The Cloud Spanner resource for which the policy is being retrieved. The format is `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e` for instance resources and `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e/databases/\u003cdatabase ID\u003e` for database resources.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:getIamPolicy",
	//   "request": {
	//     "$ref": "GetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.list":

type ProjectsInstancesDatabasesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists Cloud Spanner databases.
func (r *ProjectsInstancesDatabasesService) List(parent string) *ProjectsInstancesDatabasesListCall {
	c := &ProjectsInstancesDatabasesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": Number of databases
// to be returned in the response. If 0 or less,
// defaults to the server's maximum allowed page size.
func (c *ProjectsInstancesDatabasesListCall) PageSize(pageSize int64) *ProjectsInstancesDatabasesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If non-empty,
// `page_token` should contain a
// next_page_token from a
// previous ListDatabasesResponse.
func (c *ProjectsInstancesDatabasesListCall) PageToken(pageToken string) *ProjectsInstancesDatabasesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesListCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesDatabasesListCall) IfNoneMatch(entityTag string) *ProjectsInstancesDatabasesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesListCall) Context(ctx context.Context) *ProjectsInstancesDatabasesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/databases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.list" call.
// Exactly one of *ListDatabasesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListDatabasesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesListCall) Do(opts ...googleapi.CallOption) (*ListDatabasesResponse, error) {
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
	ret := &ListDatabasesResponse{
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
	//   "description": "Lists Cloud Spanner databases.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.databases.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Number of databases to be returned in the response. If 0 or less,\ndefaults to the server's maximum allowed page size.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "If non-empty, `page_token` should contain a\nnext_page_token from a\nprevious ListDatabasesResponse.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The instance whose databases should be listed.\nValues are of the form `projects/\u003cproject\u003e/instances/\u003cinstance\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/databases",
	//   "response": {
	//     "$ref": "ListDatabasesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsInstancesDatabasesListCall) Pages(ctx context.Context, f func(*ListDatabasesResponse) error) error {
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

// method id "spanner.projects.instances.databases.setIamPolicy":

type ProjectsInstancesDatabasesSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy on a database resource.
// Replaces any
// existing policy.
//
// Authorization requires `spanner.databases.setIamPolicy` permission
// on
// resource.
func (r *ProjectsInstancesDatabasesService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ProjectsInstancesDatabasesSetIamPolicyCall {
	c := &ProjectsInstancesDatabasesSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSetIamPolicyCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.setiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Sets the access control policy on a database resource. Replaces any\nexisting policy.\n\nAuthorization requires `spanner.databases.setIamPolicy` permission on\nresource.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The Cloud Spanner resource for which the policy is being set. The format is `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e` for instance resources and `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e/databases/\u003cdatabase ID\u003e` for databases resources.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:setIamPolicy",
	//   "request": {
	//     "$ref": "SetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.testIamPermissions":

type ProjectsInstancesDatabasesTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that the caller has on the
// specified database resource.
//
// Attempting this RPC on a non-existent Cloud Spanner database will
// result in
// a NOT_FOUND error if the user has `spanner.databases.list` permission
// on
// the containing Cloud Spanner instance. Otherwise returns an empty set
// of
// permissions.
func (r *ProjectsInstancesDatabasesService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ProjectsInstancesDatabasesTestIamPermissionsCall {
	c := &ProjectsInstancesDatabasesTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesTestIamPermissionsCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesTestIamPermissionsCall) Context(ctx context.Context) *ProjectsInstancesDatabasesTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testiampermissionsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	ret := &TestIamPermissionsResponse{
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
	//   "description": "Returns permissions that the caller has on the specified database resource.\n\nAttempting this RPC on a non-existent Cloud Spanner database will result in\na NOT_FOUND error if the user has `spanner.databases.list` permission on\nthe containing Cloud Spanner instance. Otherwise returns an empty set of\npermissions.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The Cloud Spanner resource for which permissions are being tested. The format is `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e` for instance resources and `projects/\u003cproject ID\u003e/instances/\u003cinstance ID\u003e/databases/\u003cdatabase ID\u003e` for database resources.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:testIamPermissions",
	//   "request": {
	//     "$ref": "TestIamPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestIamPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.updateDdl":

type ProjectsInstancesDatabasesUpdateDdlCall struct {
	s                        *Service
	database                 string
	updatedatabaseddlrequest *UpdateDatabaseDdlRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// UpdateDdl: Updates the schema of a Cloud Spanner database
// by
// creating/altering/dropping tables, columns, indexes, etc. The
// returned
// long-running operation will have a name of
// the format `<database_name>/operations/<operation_id>` and can be
// used to
// track execution of the schema change(s). The
// metadata field type is
// UpdateDatabaseDdlMetadata.  The operation has no response.
func (r *ProjectsInstancesDatabasesService) UpdateDdl(database string, updatedatabaseddlrequest *UpdateDatabaseDdlRequest) *ProjectsInstancesDatabasesUpdateDdlCall {
	c := &ProjectsInstancesDatabasesUpdateDdlCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.database = database
	c.updatedatabaseddlrequest = updatedatabaseddlrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesUpdateDdlCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesUpdateDdlCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesUpdateDdlCall) Context(ctx context.Context) *ProjectsInstancesDatabasesUpdateDdlCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesUpdateDdlCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesUpdateDdlCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updatedatabaseddlrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+database}/ddl")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.updateDdl" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesUpdateDdlCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Updates the schema of a Cloud Spanner database by\ncreating/altering/dropping tables, columns, indexes, etc. The returned\nlong-running operation will have a name of\nthe format `\u003cdatabase_name\u003e/operations/\u003coperation_id\u003e` and can be used to\ntrack execution of the schema change(s). The\nmetadata field type is\nUpdateDatabaseDdlMetadata.  The operation has no response.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/ddl",
	//   "httpMethod": "PATCH",
	//   "id": "spanner.projects.instances.databases.updateDdl",
	//   "parameterOrder": [
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Required. The database to update.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+database}/ddl",
	//   "request": {
	//     "$ref": "UpdateDatabaseDdlRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.operations.cancel":

type ProjectsInstancesDatabasesOperationsCancelCall struct {
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
func (r *ProjectsInstancesDatabasesOperationsService) Cancel(name string) *ProjectsInstancesDatabasesOperationsCancelCall {
	c := &ProjectsInstancesDatabasesOperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesOperationsCancelCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesOperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesOperationsCancelCall) Context(ctx context.Context) *ProjectsInstancesDatabasesOperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesOperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesOperationsCancelCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.databases.operations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesOperationsCancelCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/operations/[^/]+$",
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
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.operations.delete":

type ProjectsInstancesDatabasesOperationsDeleteCall struct {
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
func (r *ProjectsInstancesDatabasesOperationsService) Delete(name string) *ProjectsInstancesDatabasesOperationsDeleteCall {
	c := &ProjectsInstancesDatabasesOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesOperationsDeleteCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesOperationsDeleteCall) Context(ctx context.Context) *ProjectsInstancesDatabasesOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.databases.operations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesOperationsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations/{operationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "spanner.projects.instances.databases.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/operations/[^/]+$",
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
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.operations.get":

type ProjectsInstancesDatabasesOperationsGetCall struct {
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
func (r *ProjectsInstancesDatabasesOperationsService) Get(name string) *ProjectsInstancesDatabasesOperationsGetCall {
	c := &ProjectsInstancesDatabasesOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesOperationsGetCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesDatabasesOperationsGetCall) IfNoneMatch(entityTag string) *ProjectsInstancesDatabasesOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesOperationsGetCall) Context(ctx context.Context) *ProjectsInstancesDatabasesOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.databases.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesOperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.databases.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.operations.list":

type ProjectsInstancesDatabasesOperationsListCall struct {
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
func (r *ProjectsInstancesDatabasesOperationsService) List(name string) *ProjectsInstancesDatabasesOperationsListCall {
	c := &ProjectsInstancesDatabasesOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": The standard list
// filter.
func (c *ProjectsInstancesDatabasesOperationsListCall) Filter(filter string) *ProjectsInstancesDatabasesOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The standard list
// page size.
func (c *ProjectsInstancesDatabasesOperationsListCall) PageSize(pageSize int64) *ProjectsInstancesDatabasesOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *ProjectsInstancesDatabasesOperationsListCall) PageToken(pageToken string) *ProjectsInstancesDatabasesOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesOperationsListCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesDatabasesOperationsListCall) IfNoneMatch(entityTag string) *ProjectsInstancesDatabasesOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesOperationsListCall) Context(ctx context.Context) *ProjectsInstancesDatabasesOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesOperationsListCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.databases.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesOperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
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
	ret := &ListOperationsResponse{
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.databases.operations.list",
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
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/operations$",
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
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsInstancesDatabasesOperationsListCall) Pages(ctx context.Context, f func(*ListOperationsResponse) error) error {
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

// method id "spanner.projects.instances.databases.sessions.beginTransaction":

type ProjectsInstancesDatabasesSessionsBeginTransactionCall struct {
	s                       *Service
	session                 string
	begintransactionrequest *BeginTransactionRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
	header_                 http.Header
}

// BeginTransaction: Begins a new transaction. This step can often be
// skipped:
// Read, ExecuteSql and
// Commit can begin a new transaction as a
// side-effect.
func (r *ProjectsInstancesDatabasesSessionsService) BeginTransaction(session string, begintransactionrequest *BeginTransactionRequest) *ProjectsInstancesDatabasesSessionsBeginTransactionCall {
	c := &ProjectsInstancesDatabasesSessionsBeginTransactionCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.session = session
	c.begintransactionrequest = begintransactionrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsBeginTransactionCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsBeginTransactionCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsBeginTransactionCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsBeginTransactionCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsBeginTransactionCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsBeginTransactionCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+session}:beginTransaction")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"session": c.session,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.beginTransaction" call.
// Exactly one of *Transaction or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Transaction.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesSessionsBeginTransactionCall) Do(opts ...googleapi.CallOption) (*Transaction, error) {
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
	ret := &Transaction{
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
	//   "description": "Begins a new transaction. This step can often be skipped:\nRead, ExecuteSql and\nCommit can begin a new transaction as a\nside-effect.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:beginTransaction",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.beginTransaction",
	//   "parameterOrder": [
	//     "session"
	//   ],
	//   "parameters": {
	//     "session": {
	//       "description": "Required. The session in which the transaction runs.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+session}:beginTransaction",
	//   "request": {
	//     "$ref": "BeginTransactionRequest"
	//   },
	//   "response": {
	//     "$ref": "Transaction"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.commit":

type ProjectsInstancesDatabasesSessionsCommitCall struct {
	s             *Service
	session       string
	commitrequest *CommitRequest
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Commit: Commits a transaction. The request includes the mutations to
// be
// applied to rows in the database.
//
// `Commit` might return an `ABORTED` error. This can occur at any
// time;
// commonly, the cause is conflicts with concurrent
// transactions. However, it can also happen for a variety of
// other
// reasons. If `Commit` returns `ABORTED`, the caller should
// re-attempt
// the transaction from the beginning, re-using the same session.
func (r *ProjectsInstancesDatabasesSessionsService) Commit(session string, commitrequest *CommitRequest) *ProjectsInstancesDatabasesSessionsCommitCall {
	c := &ProjectsInstancesDatabasesSessionsCommitCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.session = session
	c.commitrequest = commitrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsCommitCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsCommitCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsCommitCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsCommitCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsCommitCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsCommitCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+session}:commit")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"session": c.session,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.commit" call.
// Exactly one of *CommitResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CommitResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesSessionsCommitCall) Do(opts ...googleapi.CallOption) (*CommitResponse, error) {
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
	//   "description": "Commits a transaction. The request includes the mutations to be\napplied to rows in the database.\n\n`Commit` might return an `ABORTED` error. This can occur at any time;\ncommonly, the cause is conflicts with concurrent\ntransactions. However, it can also happen for a variety of other\nreasons. If `Commit` returns `ABORTED`, the caller should re-attempt\nthe transaction from the beginning, re-using the same session.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:commit",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.commit",
	//   "parameterOrder": [
	//     "session"
	//   ],
	//   "parameters": {
	//     "session": {
	//       "description": "Required. The session in which the transaction to be committed is running.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+session}:commit",
	//   "request": {
	//     "$ref": "CommitRequest"
	//   },
	//   "response": {
	//     "$ref": "CommitResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.create":

type ProjectsInstancesDatabasesSessionsCreateCall struct {
	s          *Service
	database   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new session. A session can be used to
// perform
// transactions that read and/or modify data in a Cloud Spanner
// database.
// Sessions are meant to be reused for many
// consecutive
// transactions.
//
// Sessions can only execute one transaction at a time. To
// execute
// multiple concurrent read-write/write-only transactions,
// create
// multiple sessions. Note that standalone reads and queries use
// a
// transaction internally, and count toward the one
// transaction
// limit.
//
// Cloud Spanner limits the number of sessions that can exist at any
// given
// time; thus, it is a good idea to delete idle and/or unneeded
// sessions.
// Aside from explicit deletes, Cloud Spanner can delete sessions for
// which no
// operations are sent for more than an hour. If a session is
// deleted,
// requests to it return `NOT_FOUND`.
//
// Idle sessions can be kept alive by sending a trivial SQL
// query
// periodically, e.g., "SELECT 1".
func (r *ProjectsInstancesDatabasesSessionsService) Create(database string) *ProjectsInstancesDatabasesSessionsCreateCall {
	c := &ProjectsInstancesDatabasesSessionsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.database = database
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsCreateCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsCreateCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+database}/sessions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.create" call.
// Exactly one of *Session or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Session.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesSessionsCreateCall) Do(opts ...googleapi.CallOption) (*Session, error) {
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
	//   "description": "Creates a new session. A session can be used to perform\ntransactions that read and/or modify data in a Cloud Spanner database.\nSessions are meant to be reused for many consecutive\ntransactions.\n\nSessions can only execute one transaction at a time. To execute\nmultiple concurrent read-write/write-only transactions, create\nmultiple sessions. Note that standalone reads and queries use a\ntransaction internally, and count toward the one transaction\nlimit.\n\nCloud Spanner limits the number of sessions that can exist at any given\ntime; thus, it is a good idea to delete idle and/or unneeded sessions.\nAside from explicit deletes, Cloud Spanner can delete sessions for which no\noperations are sent for more than an hour. If a session is deleted,\nrequests to it return `NOT_FOUND`.\n\nIdle sessions can be kept alive by sending a trivial SQL query\nperiodically, e.g., `\"SELECT 1\"`.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.create",
	//   "parameterOrder": [
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Required. The database in which the new session is created.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+database}/sessions",
	//   "response": {
	//     "$ref": "Session"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.delete":

type ProjectsInstancesDatabasesSessionsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Ends a session, releasing server resources associated with
// it.
func (r *ProjectsInstancesDatabasesSessionsService) Delete(name string) *ProjectsInstancesDatabasesSessionsDeleteCall {
	c := &ProjectsInstancesDatabasesSessionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsDeleteCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsDeleteCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.databases.sessions.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesSessionsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Ends a session, releasing server resources associated with it.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}",
	//   "httpMethod": "DELETE",
	//   "id": "spanner.projects.instances.databases.sessions.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The name of the session to delete.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
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
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.executeSql":

type ProjectsInstancesDatabasesSessionsExecuteSqlCall struct {
	s                 *Service
	session           string
	executesqlrequest *ExecuteSqlRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// ExecuteSql: Executes an SQL query, returning all rows in a single
// reply. This
// method cannot be used to return a result set larger than 10 MiB;
// if the query yields more data than that, the query fails with
// a `FAILED_PRECONDITION` error.
//
// Queries inside read-write transactions might return `ABORTED`.
// If
// this occurs, the application should restart the transaction from
// the beginning. See Transaction for more details.
//
// Larger result sets can be fetched in streaming fashion by
// calling
// ExecuteStreamingSql instead.
func (r *ProjectsInstancesDatabasesSessionsService) ExecuteSql(session string, executesqlrequest *ExecuteSqlRequest) *ProjectsInstancesDatabasesSessionsExecuteSqlCall {
	c := &ProjectsInstancesDatabasesSessionsExecuteSqlCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.session = session
	c.executesqlrequest = executesqlrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsExecuteSqlCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsExecuteSqlCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsExecuteSqlCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsExecuteSqlCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsExecuteSqlCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsExecuteSqlCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.executesqlrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+session}:executeSql")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"session": c.session,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.executeSql" call.
// Exactly one of *ResultSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ResultSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesSessionsExecuteSqlCall) Do(opts ...googleapi.CallOption) (*ResultSet, error) {
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
	ret := &ResultSet{
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
	//   "description": "Executes an SQL query, returning all rows in a single reply. This\nmethod cannot be used to return a result set larger than 10 MiB;\nif the query yields more data than that, the query fails with\na `FAILED_PRECONDITION` error.\n\nQueries inside read-write transactions might return `ABORTED`. If\nthis occurs, the application should restart the transaction from\nthe beginning. See Transaction for more details.\n\nLarger result sets can be fetched in streaming fashion by calling\nExecuteStreamingSql instead.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:executeSql",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.executeSql",
	//   "parameterOrder": [
	//     "session"
	//   ],
	//   "parameters": {
	//     "session": {
	//       "description": "Required. The session in which the SQL query should be performed.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+session}:executeSql",
	//   "request": {
	//     "$ref": "ExecuteSqlRequest"
	//   },
	//   "response": {
	//     "$ref": "ResultSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.executeStreamingSql":

type ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall struct {
	s                 *Service
	session           string
	executesqlrequest *ExecuteSqlRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// ExecuteStreamingSql: Like ExecuteSql, except returns the result
// set as a stream. Unlike ExecuteSql, there
// is no limit on the size of the returned result set. However,
// no
// individual row in the result set can exceed 100 MiB, and no
// column value can exceed 10 MiB.
func (r *ProjectsInstancesDatabasesSessionsService) ExecuteStreamingSql(session string, executesqlrequest *ExecuteSqlRequest) *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall {
	c := &ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.session = session
	c.executesqlrequest = executesqlrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.executesqlrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+session}:executeStreamingSql")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"session": c.session,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.executeStreamingSql" call.
// Exactly one of *PartialResultSet or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PartialResultSet.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesSessionsExecuteStreamingSqlCall) Do(opts ...googleapi.CallOption) (*PartialResultSet, error) {
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
	ret := &PartialResultSet{
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
	//   "description": "Like ExecuteSql, except returns the result\nset as a stream. Unlike ExecuteSql, there\nis no limit on the size of the returned result set. However, no\nindividual row in the result set can exceed 100 MiB, and no\ncolumn value can exceed 10 MiB.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:executeStreamingSql",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.executeStreamingSql",
	//   "parameterOrder": [
	//     "session"
	//   ],
	//   "parameters": {
	//     "session": {
	//       "description": "Required. The session in which the SQL query should be performed.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+session}:executeStreamingSql",
	//   "request": {
	//     "$ref": "ExecuteSqlRequest"
	//   },
	//   "response": {
	//     "$ref": "PartialResultSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.get":

type ProjectsInstancesDatabasesSessionsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a session. Returns `NOT_FOUND` if the session does not
// exist.
// This is mainly useful for determining whether a session is
// still
// alive.
func (r *ProjectsInstancesDatabasesSessionsService) Get(name string) *ProjectsInstancesDatabasesSessionsGetCall {
	c := &ProjectsInstancesDatabasesSessionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsGetCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesDatabasesSessionsGetCall) IfNoneMatch(entityTag string) *ProjectsInstancesDatabasesSessionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsGetCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.databases.sessions.get" call.
// Exactly one of *Session or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Session.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesSessionsGetCall) Do(opts ...googleapi.CallOption) (*Session, error) {
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
	//   "description": "Gets a session. Returns `NOT_FOUND` if the session does not exist.\nThis is mainly useful for determining whether a session is still\nalive.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.databases.sessions.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The name of the session to retrieve.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Session"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.read":

type ProjectsInstancesDatabasesSessionsReadCall struct {
	s           *Service
	session     string
	readrequest *ReadRequest
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Read: Reads rows from the database using key lookups and scans, as
// a
// simple key/value style alternative to
// ExecuteSql.  This method cannot be used to
// return a result set larger than 10 MiB; if the read matches more
// data than that, the read fails with a
// `FAILED_PRECONDITION`
// error.
//
// Reads inside read-write transactions might return `ABORTED`. If
// this occurs, the application should restart the transaction from
// the beginning. See Transaction for more details.
//
// Larger result sets can be yielded in streaming fashion by
// calling
// StreamingRead instead.
func (r *ProjectsInstancesDatabasesSessionsService) Read(session string, readrequest *ReadRequest) *ProjectsInstancesDatabasesSessionsReadCall {
	c := &ProjectsInstancesDatabasesSessionsReadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.session = session
	c.readrequest = readrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsReadCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsReadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsReadCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsReadCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsReadCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsReadCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.readrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+session}:read")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"session": c.session,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.read" call.
// Exactly one of *ResultSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ResultSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesSessionsReadCall) Do(opts ...googleapi.CallOption) (*ResultSet, error) {
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
	ret := &ResultSet{
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
	//   "description": "Reads rows from the database using key lookups and scans, as a\nsimple key/value style alternative to\nExecuteSql.  This method cannot be used to\nreturn a result set larger than 10 MiB; if the read matches more\ndata than that, the read fails with a `FAILED_PRECONDITION`\nerror.\n\nReads inside read-write transactions might return `ABORTED`. If\nthis occurs, the application should restart the transaction from\nthe beginning. See Transaction for more details.\n\nLarger result sets can be yielded in streaming fashion by calling\nStreamingRead instead.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:read",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.read",
	//   "parameterOrder": [
	//     "session"
	//   ],
	//   "parameters": {
	//     "session": {
	//       "description": "Required. The session in which the read should be performed.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+session}:read",
	//   "request": {
	//     "$ref": "ReadRequest"
	//   },
	//   "response": {
	//     "$ref": "ResultSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.rollback":

type ProjectsInstancesDatabasesSessionsRollbackCall struct {
	s               *Service
	session         string
	rollbackrequest *RollbackRequest
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Rollback: Rolls back a transaction, releasing any locks it holds. It
// is a good
// idea to call this for any transaction that includes one or more
// Read or ExecuteSql requests and
// ultimately decides not to commit.
//
// `Rollback` returns `OK` if it successfully aborts the transaction,
// the
// transaction was already aborted, or the transaction is not
// found. `Rollback` never returns `ABORTED`.
func (r *ProjectsInstancesDatabasesSessionsService) Rollback(session string, rollbackrequest *RollbackRequest) *ProjectsInstancesDatabasesSessionsRollbackCall {
	c := &ProjectsInstancesDatabasesSessionsRollbackCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.session = session
	c.rollbackrequest = rollbackrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsRollbackCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsRollbackCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsRollbackCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsRollbackCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsRollbackCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsRollbackCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+session}:rollback")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"session": c.session,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.rollback" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesDatabasesSessionsRollbackCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Rolls back a transaction, releasing any locks it holds. It is a good\nidea to call this for any transaction that includes one or more\nRead or ExecuteSql requests and\nultimately decides not to commit.\n\n`Rollback` returns `OK` if it successfully aborts the transaction, the\ntransaction was already aborted, or the transaction is not\nfound. `Rollback` never returns `ABORTED`.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:rollback",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.rollback",
	//   "parameterOrder": [
	//     "session"
	//   ],
	//   "parameters": {
	//     "session": {
	//       "description": "Required. The session in which the transaction to roll back is running.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+session}:rollback",
	//   "request": {
	//     "$ref": "RollbackRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.databases.sessions.streamingRead":

type ProjectsInstancesDatabasesSessionsStreamingReadCall struct {
	s           *Service
	session     string
	readrequest *ReadRequest
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// StreamingRead: Like Read, except returns the result set as a
// stream. Unlike Read, there is no limit on the
// size of the returned result set. However, no individual row in
// the result set can exceed 100 MiB, and no column value can exceed
// 10 MiB.
func (r *ProjectsInstancesDatabasesSessionsService) StreamingRead(session string, readrequest *ReadRequest) *ProjectsInstancesDatabasesSessionsStreamingReadCall {
	c := &ProjectsInstancesDatabasesSessionsStreamingReadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.session = session
	c.readrequest = readrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesDatabasesSessionsStreamingReadCall) Fields(s ...googleapi.Field) *ProjectsInstancesDatabasesSessionsStreamingReadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesDatabasesSessionsStreamingReadCall) Context(ctx context.Context) *ProjectsInstancesDatabasesSessionsStreamingReadCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesDatabasesSessionsStreamingReadCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesDatabasesSessionsStreamingReadCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.readrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+session}:streamingRead")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"session": c.session,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "spanner.projects.instances.databases.sessions.streamingRead" call.
// Exactly one of *PartialResultSet or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PartialResultSet.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesDatabasesSessionsStreamingReadCall) Do(opts ...googleapi.CallOption) (*PartialResultSet, error) {
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
	ret := &PartialResultSet{
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
	//   "description": "Like Read, except returns the result set as a\nstream. Unlike Read, there is no limit on the\nsize of the returned result set. However, no individual row in\nthe result set can exceed 100 MiB, and no column value can exceed\n10 MiB.",
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:streamingRead",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.databases.sessions.streamingRead",
	//   "parameterOrder": [
	//     "session"
	//   ],
	//   "parameters": {
	//     "session": {
	//       "description": "Required. The session in which the read should be performed.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/databases/[^/]+/sessions/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+session}:streamingRead",
	//   "request": {
	//     "$ref": "ReadRequest"
	//   },
	//   "response": {
	//     "$ref": "PartialResultSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.data"
	//   ]
	// }

}

// method id "spanner.projects.instances.operations.cancel":

type ProjectsInstancesOperationsCancelCall struct {
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
func (r *ProjectsInstancesOperationsService) Cancel(name string) *ProjectsInstancesOperationsCancelCall {
	c := &ProjectsInstancesOperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesOperationsCancelCall) Fields(s ...googleapi.Field) *ProjectsInstancesOperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesOperationsCancelCall) Context(ctx context.Context) *ProjectsInstancesOperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesOperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesOperationsCancelCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.operations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesOperationsCancelCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "spanner.projects.instances.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/operations/[^/]+$",
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
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.operations.delete":

type ProjectsInstancesOperationsDeleteCall struct {
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
func (r *ProjectsInstancesOperationsService) Delete(name string) *ProjectsInstancesOperationsDeleteCall {
	c := &ProjectsInstancesOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesOperationsDeleteCall) Fields(s ...googleapi.Field) *ProjectsInstancesOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesOperationsDeleteCall) Context(ctx context.Context) *ProjectsInstancesOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.operations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsInstancesOperationsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/operations/{operationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "spanner.projects.instances.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/operations/[^/]+$",
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
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.operations.get":

type ProjectsInstancesOperationsGetCall struct {
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
func (r *ProjectsInstancesOperationsService) Get(name string) *ProjectsInstancesOperationsGetCall {
	c := &ProjectsInstancesOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesOperationsGetCall) Fields(s ...googleapi.Field) *ProjectsInstancesOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesOperationsGetCall) IfNoneMatch(entityTag string) *ProjectsInstancesOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesOperationsGetCall) Context(ctx context.Context) *ProjectsInstancesOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsInstancesOperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/instances/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// method id "spanner.projects.instances.operations.list":

type ProjectsInstancesOperationsListCall struct {
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
func (r *ProjectsInstancesOperationsService) List(name string) *ProjectsInstancesOperationsListCall {
	c := &ProjectsInstancesOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": The standard list
// filter.
func (c *ProjectsInstancesOperationsListCall) Filter(filter string) *ProjectsInstancesOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The standard list
// page size.
func (c *ProjectsInstancesOperationsListCall) PageSize(pageSize int64) *ProjectsInstancesOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *ProjectsInstancesOperationsListCall) PageToken(pageToken string) *ProjectsInstancesOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsInstancesOperationsListCall) Fields(s ...googleapi.Field) *ProjectsInstancesOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsInstancesOperationsListCall) IfNoneMatch(entityTag string) *ProjectsInstancesOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsInstancesOperationsListCall) Context(ctx context.Context) *ProjectsInstancesOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsInstancesOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsInstancesOperationsListCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "spanner.projects.instances.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsInstancesOperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
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
	ret := &ListOperationsResponse{
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
	//   "flatPath": "v1/projects/{projectsId}/instances/{instancesId}/operations",
	//   "httpMethod": "GET",
	//   "id": "spanner.projects.instances.operations.list",
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
	//       "pattern": "^projects/[^/]+/instances/[^/]+/operations$",
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
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/spanner.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsInstancesOperationsListCall) Pages(ctx context.Context, f func(*ListOperationsResponse) error) error {
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
