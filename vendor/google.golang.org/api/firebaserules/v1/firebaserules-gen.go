// Package firebaserules provides access to the Firebase Rules API.
//
// See https://firebase.google.com/docs/storage/security
//
// Usage example:
//
//   import "google.golang.org/api/firebaserules/v1"
//   ...
//   firebaserulesService, err := firebaserules.New(oauthHttpClient)
package firebaserules // import "google.golang.org/api/firebaserules/v1"

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

const apiId = "firebaserules:v1"
const apiName = "firebaserules"
const apiVersion = "v1"
const basePath = "https://firebaserules.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View and administer all your Firebase data and settings
	FirebaseScope = "https://www.googleapis.com/auth/firebase"

	// View all your Firebase data and settings
	FirebaseReadonlyScope = "https://www.googleapis.com/auth/firebase.readonly"
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
	rs.Releases = NewProjectsReleasesService(s)
	rs.Rulesets = NewProjectsRulesetsService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Releases *ProjectsReleasesService

	Rulesets *ProjectsRulesetsService
}

func NewProjectsReleasesService(s *Service) *ProjectsReleasesService {
	rs := &ProjectsReleasesService{s: s}
	return rs
}

type ProjectsReleasesService struct {
	s *Service
}

func NewProjectsRulesetsService(s *Service) *ProjectsRulesetsService {
	rs := &ProjectsRulesetsService{s: s}
	return rs
}

type ProjectsRulesetsService struct {
	s *Service
}

// Arg: Arg matchers for the mock function.
type Arg struct {
	// AnyValue: Argument matches any value provided.
	AnyValue *Empty `json:"anyValue,omitempty"`

	// ExactValue: Argument exactly matches value provided.
	ExactValue interface{} `json:"exactValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnyValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnyValue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Arg) MarshalJSON() ([]byte, error) {
	type noMethod Arg
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

// File: `File` containing source content.
type File struct {
	// Content: Textual Content.
	Content string `json:"content,omitempty"`

	// Fingerprint: Fingerprint (e.g. github sha) associated with the
	// `File`.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Name: File name.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Content") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Content") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *File) MarshalJSON() ([]byte, error) {
	type noMethod File
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FunctionCall: Represents a service-defined function call that was
// invoked during test
// execution.
type FunctionCall struct {
	// Args: The arguments that were provided to the function.
	Args []interface{} `json:"args,omitempty"`

	// Function: Name of the function invoked.
	Function string `json:"function,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Args") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Args") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FunctionCall) MarshalJSON() ([]byte, error) {
	type noMethod FunctionCall
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FunctionMock: Mock function definition.
//
// Mocks must refer to a function declared by the target service. The
// type of
// the function args and result will be inferred at test time. If either
// the
// arg or result values are not compatible with function type
// declaration, the
// request will be considered invalid.
//
// More than one `FunctionMock` may be provided for a given function
// name so
// long as the `Arg` matchers are distinct. There may be only one
// function
// for a given overload where all `Arg` values are `Arg.any_value`.
type FunctionMock struct {
	// Args: The list of `Arg` values to match. The order in which the
	// arguments are
	// provided is the order in which they must appear in the
	// function
	// invocation.
	Args []*Arg `json:"args,omitempty"`

	// Function: The name of the function.
	//
	// The function name must match one provided by a service declaration.
	Function string `json:"function,omitempty"`

	// Result: The mock result of the function call.
	Result *Result `json:"result,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Args") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Args") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FunctionMock) MarshalJSON() ([]byte, error) {
	type noMethod FunctionMock
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Issue: Issues include warnings, errors, and deprecation notices.
type Issue struct {
	// Description: Short error description.
	Description string `json:"description,omitempty"`

	// Severity: The severity of the issue.
	//
	// Possible values:
	//   "SEVERITY_UNSPECIFIED" - An unspecified severity.
	//   "DEPRECATION" - Deprecation issue for statements and method that
	// may no longer be
	// supported or maintained.
	//   "WARNING" - Warnings such as: unused variables.
	//   "ERROR" - Errors such as: unmatched curly braces or variable
	// redefinition.
	Severity string `json:"severity,omitempty"`

	// SourcePosition: Position of the issue in the `Source`.
	SourcePosition *SourcePosition `json:"sourcePosition,omitempty"`

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

func (s *Issue) MarshalJSON() ([]byte, error) {
	type noMethod Issue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListReleasesResponse: The response for
// FirebaseRulesService.ListReleases.
type ListReleasesResponse struct {
	// NextPageToken: The pagination token to retrieve the next page of
	// results. If the value is
	// empty, no further results remain.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Releases: List of `Release` instances.
	Releases []*Release `json:"releases,omitempty"`

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

func (s *ListReleasesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListReleasesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListRulesetsResponse: The response for
// FirebaseRulesService.ListRulesets.
type ListRulesetsResponse struct {
	// NextPageToken: The pagination token to retrieve the next page of
	// results. If the value is
	// empty, no further results remain.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Rulesets: List of `Ruleset` instances.
	Rulesets []*Ruleset `json:"rulesets,omitempty"`

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

func (s *ListRulesetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListRulesetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Release: `Release` is a named reference to a `Ruleset`. Once a
// `Release` refers to a
// `Ruleset`, rules-enabled services will be able to enforce the
// `Ruleset`.
type Release struct {
	// CreateTime: Time the release was created.
	// Output only.
	CreateTime string `json:"createTime,omitempty"`

	// Name: Resource name for the `Release`.
	//
	// `Release` names may be structured `app1/prod/v2` or flat
	// `app1_prod_v2`
	// which affords developers a great deal of flexibility in mapping the
	// name
	// to the style that best fits their existing development practices.
	// For
	// example, a name could refer to an environment, an app, a version, or
	// some
	// combination of three.
	//
	// In the table below, for the project name `projects/foo`, the
	// following
	// relative release paths show how flat and structured names might be
	// chosen
	// to match a desired development / deployment strategy.
	//
	// Use Case     | Flat Name           | Structured
	// Name
	// -------------|---------------------|----------------
	// Environments
	//  | releases/qa         | releases/qa
	// Apps         | releases/app1_qa    | releases/app1/qa
	// Versions     | releases/app1_v2_qa | releases/app1/v2/qa
	//
	// The delimiter between the release name path elements can be almost
	// anything
	// and it should work equally well with the release name list filter,
	// but in
	// many ways the structured paths provide a clearer picture of
	// the
	// relationship between `Release` instances.
	//
	// Format: `projects/{project_id}/releases/{release_id}`
	Name string `json:"name,omitempty"`

	// RulesetName: Name of the `Ruleset` referred to by this `Release`. The
	// `Ruleset` must
	// exist the `Release` to be created.
	RulesetName string `json:"rulesetName,omitempty"`

	// UpdateTime: Time the release was updated.
	// Output only.
	UpdateTime string `json:"updateTime,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *Release) MarshalJSON() ([]byte, error) {
	type noMethod Release
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Result: Possible result values from the function mock invocation.
type Result struct {
	// Undefined: The result is undefined, meaning the result could not be
	// computed.
	Undefined *Empty `json:"undefined,omitempty"`

	// Value: The result is an actual value. The type of the value must
	// match that
	// of the type declared by the service.
	Value interface{} `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Undefined") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Undefined") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Result) MarshalJSON() ([]byte, error) {
	type noMethod Result
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Ruleset: `Ruleset` is an immutable copy of `Source` with a globally
// unique identifier
// and a creation time.
type Ruleset struct {
	// CreateTime: Time the `Ruleset` was created.
	// Output only.
	CreateTime string `json:"createTime,omitempty"`

	// Name: Name of the `Ruleset`. The ruleset_id is auto generated by the
	// service.
	// Format: `projects/{project_id}/rulesets/{ruleset_id}`
	// Output only.
	Name string `json:"name,omitempty"`

	// Source: `Source` for the `Ruleset`.
	Source *Source `json:"source,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *Ruleset) MarshalJSON() ([]byte, error) {
	type noMethod Ruleset
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Source: `Source` is one or more `File` messages comprising a logical
// set of rules.
type Source struct {
	// Files: `File` set constituting the `Source` bundle.
	Files []*File `json:"files,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Files") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Files") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Source) MarshalJSON() ([]byte, error) {
	type noMethod Source
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SourcePosition: Position in the `Source` content including its line,
// column number, and an
// index of the `File` in the `Source` message. Used for debug purposes.
type SourcePosition struct {
	// Column: First column on the source line associated with the source
	// fragment.
	Column int64 `json:"column,omitempty"`

	// FileName: Name of the `File`.
	FileName string `json:"fileName,omitempty"`

	// Line: Line number of the source fragment. 1-based.
	Line int64 `json:"line,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Column") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Column") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SourcePosition) MarshalJSON() ([]byte, error) {
	type noMethod SourcePosition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestCase: `TestCase` messages provide the request context and an
// expectation as to
// whether the given context will be allowed or denied. Test cases may
// specify
// the `request`, `resource`, and `function_mocks` to mock a function
// call to
// a service-provided function.
//
// The `request` object represents context present at request-time.
//
// The `resource` is the value of the target resource as it appears
// in
// persistent storage before the request is executed.
type TestCase struct {
	// Expectation: Test expectation.
	//
	// Possible values:
	//   "EXPECTATION_UNSPECIFIED" - Unspecified expectation.
	//   "ALLOW" - Expect an allowed result.
	//   "DENY" - Expect a denied result.
	Expectation string `json:"expectation,omitempty"`

	// FunctionMocks: Optional function mocks for service-defined functions.
	// If not set, any
	// service defined function is expected to return an error, which may or
	// may
	// not influence the test outcome.
	FunctionMocks []*FunctionMock `json:"functionMocks,omitempty"`

	// Request: Request context.
	//
	// The exact format of the request context is service-dependent. See
	// the
	// appropriate service documentation for information about the
	// supported
	// fields and types on the request. Minimally, all services support
	// the
	// following fields and types:
	//
	// Request field  | Type
	// ---------------|-----------------
	// auth.uid       | `string`
	// auth.token     | `map<string, string>`
	// headers        | `map<string, string>`
	// method         | `string`
	// params         | `map<string, string>`
	// path           | `string`
	// time           | `google.protobuf.Timestamp`
	//
	// If the request value is not well-formed for the service, the request
	// will
	// be rejected as an invalid argument.
	Request interface{} `json:"request,omitempty"`

	// Resource: Optional resource value as it appears in persistent storage
	// before the
	// request is fulfilled.
	//
	// The resource type depends on the `request.path` value.
	Resource interface{} `json:"resource,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Expectation") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Expectation") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestCase) MarshalJSON() ([]byte, error) {
	type noMethod TestCase
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestResult: Test result message containing the state of the test as
// well as a
// description and source position for test failures.
type TestResult struct {
	// DebugMessages: Debug messages related to test execution issues
	// encountered during
	// evaluation.
	//
	// Debug messages may be related to too many or too few invocations
	// of
	// function mocks or to runtime errors that occur during
	// evaluation.
	//
	// For example: ```Unable to read variable [name: "resource"]```
	DebugMessages []string `json:"debugMessages,omitempty"`

	// ErrorPosition: Position in the `Source` or `Ruleset` where the
	// principle runtime error
	// occurs.
	//
	// Evaluation of an expression may result in an error. Rules are deny
	// by
	// default, so a `DENY` expectation when an error is generated is
	// valid.
	// When there is a `DENY` with an error, the `SourcePosition` is
	// returned.
	//
	// E.g. `error_position { line: 19 column: 37 }`
	ErrorPosition *SourcePosition `json:"errorPosition,omitempty"`

	// FunctionCalls: The set of function calls made to service-defined
	// methods.
	//
	// Function calls are included in the order in which they are
	// encountered
	// during evaluation, are provided for both mocked and unmocked
	// functions,
	// and included on the response regardless of the test `state`.
	FunctionCalls []*FunctionCall `json:"functionCalls,omitempty"`

	// State: State of the test.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED" - Test state is not set.
	//   "SUCCESS" - Test is a success.
	//   "FAILURE" - Test is a failure.
	State string `json:"state,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DebugMessages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DebugMessages") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestResult) MarshalJSON() ([]byte, error) {
	type noMethod TestResult
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestRulesetRequest: The request for FirebaseRulesService.TestRuleset.
type TestRulesetRequest struct {
	// Source: Optional `Source` to be checked for correctness.
	//
	// This field must not be set when the resource name refers to a
	// `Ruleset`.
	Source *Source `json:"source,omitempty"`

	// TestSuite: Inline `TestSuite` to run.
	TestSuite *TestSuite `json:"testSuite,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Source") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Source") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestRulesetRequest) MarshalJSON() ([]byte, error) {
	type noMethod TestRulesetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestRulesetResponse: The response for
// FirebaseRulesService.TestRuleset.
type TestRulesetResponse struct {
	// Issues: Syntactic and semantic `Source` issues of varying severity.
	// Issues of
	// `ERROR` severity will prevent tests from executing.
	Issues []*Issue `json:"issues,omitempty"`

	// TestResults: The set of test results given the test cases in the
	// `TestSuite`.
	// The results will appear in the same order as the test cases appear in
	// the
	// `TestSuite`.
	TestResults []*TestResult `json:"testResults,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Issues") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Issues") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestRulesetResponse) MarshalJSON() ([]byte, error) {
	type noMethod TestRulesetResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestSuite: `TestSuite` is a collection of `TestCase` instances that
// validate the logical
// correctness of a `Ruleset`. The `TestSuite` may be referenced in-line
// within
// a `TestRuleset` invocation or as part of a `Release` object as a
// pre-release
// check.
type TestSuite struct {
	// TestCases: Collection of test cases associated with the `TestSuite`.
	TestCases []*TestCase `json:"testCases,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TestCases") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TestCases") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestSuite) MarshalJSON() ([]byte, error) {
	type noMethod TestSuite
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "firebaserules.projects.test":

type ProjectsTestCall struct {
	s                  *Service
	name               string
	testrulesetrequest *TestRulesetRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// Test: Test `Source` for syntactic and semantic correctness. Issues
// present, if
// any, will be returned to the caller with a description, severity,
// and
// source location.
//
// The test method may be executed with `Source` or a `Ruleset`
// name.
// Passing `Source` is useful for unit testing new rules. Passing a
// `Ruleset`
// name is useful for regression testing an existing rule.
//
// The following is an example of `Source` that permits users to upload
// images
// to a bucket bearing their user id and matching the correct
// metadata:
//
// _*Example*_
//
//     // Users are allowed to subscribe and unsubscribe to the blog.
//     service firebase.storage {
//       match /users/{userId}/images/{imageName} {
//           allow write: if userId == request.auth.uid
//               && (imageName.matches('*.png$')
//               || imageName.matches('*.jpg$'))
//               && resource.mimeType.matches('^image/')
//       }
//     }
func (r *ProjectsService) Test(name string, testrulesetrequest *TestRulesetRequest) *ProjectsTestCall {
	c := &ProjectsTestCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.testrulesetrequest = testrulesetrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTestCall) Fields(s ...googleapi.Field) *ProjectsTestCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTestCall) Context(ctx context.Context) *ProjectsTestCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsTestCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsTestCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testrulesetrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:test")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "firebaserules.projects.test" call.
// Exactly one of *TestRulesetResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TestRulesetResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsTestCall) Do(opts ...googleapi.CallOption) (*TestRulesetResponse, error) {
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
	ret := &TestRulesetResponse{
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
	//   "description": "Test `Source` for syntactic and semantic correctness. Issues present, if\nany, will be returned to the caller with a description, severity, and\nsource location.\n\nThe test method may be executed with `Source` or a `Ruleset` name.\nPassing `Source` is useful for unit testing new rules. Passing a `Ruleset`\nname is useful for regression testing an existing rule.\n\nThe following is an example of `Source` that permits users to upload images\nto a bucket bearing their user id and matching the correct metadata:\n\n_*Example*_\n\n    // Users are allowed to subscribe and unsubscribe to the blog.\n    service firebase.storage {\n      match /users/{userId}/images/{imageName} {\n          allow write: if userId == request.auth.uid\n              \u0026\u0026 (imageName.matches('*.png$')\n              || imageName.matches('*.jpg$'))\n              \u0026\u0026 resource.mimeType.matches('^image/')\n      }\n    }",
	//   "flatPath": "v1/projects/{projectsId}:test",
	//   "httpMethod": "POST",
	//   "id": "firebaserules.projects.test",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Tests may either provide `source` or a `Ruleset` resource name.\n\nFor tests against `source`, the resource name must refer to the project:\nFormat: `projects/{project_id}`\n\nFor tests against a `Ruleset`, this must be the `Ruleset` resource name:\nFormat: `projects/{project_id}/rulesets/{ruleset_id}`",
	//       "location": "path",
	//       "pattern": "^projects/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:test",
	//   "request": {
	//     "$ref": "TestRulesetRequest"
	//   },
	//   "response": {
	//     "$ref": "TestRulesetResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase",
	//     "https://www.googleapis.com/auth/firebase.readonly"
	//   ]
	// }

}

// method id "firebaserules.projects.releases.create":

type ProjectsReleasesCreateCall struct {
	s          *Service
	name       string
	release    *Release
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Create a `Release`.
//
// Release names should reflect the developer's deployment practices.
// For
// example, the release name may include the environment name,
// application
// name, application version, or any other name meaningful to the
// developer.
// Once a `Release` refers to a `Ruleset`, the rules can be enforced
// by
// Firebase Rules-enabled services.
//
// More than one `Release` may be 'live' concurrently. Consider the
// following
// three `Release` names for `projects/foo` and the `Ruleset` to which
// they
// refer.
//
// Release Name                    | Ruleset
// Name
// --------------------------------|-------------
// projects/foo/relea
// ses/prod      |
// projects/foo/rulesets/uuid123
// projects/foo/releases/prod/beta |
// projects/foo/rulesets/uuid123
// projects/foo/releases/prod/v23  | projects/foo/rulesets/uuid456
//
// The table reflects the `Ruleset` rollout in progress. The `prod`
// and
// `prod/beta` releases refer to the same `Ruleset`. However,
// `prod/v23`
// refers to a new `Ruleset`. The `Ruleset` reference for a `Release`
// may be
// updated using the UpdateRelease method.
func (r *ProjectsReleasesService) Create(name string, release *Release) *ProjectsReleasesCreateCall {
	c := &ProjectsReleasesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.release = release
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReleasesCreateCall) Fields(s ...googleapi.Field) *ProjectsReleasesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReleasesCreateCall) Context(ctx context.Context) *ProjectsReleasesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReleasesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReleasesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.release)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/releases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "firebaserules.projects.releases.create" call.
// Exactly one of *Release or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Release.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsReleasesCreateCall) Do(opts ...googleapi.CallOption) (*Release, error) {
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
	ret := &Release{
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
	//   "description": "Create a `Release`.\n\nRelease names should reflect the developer's deployment practices. For\nexample, the release name may include the environment name, application\nname, application version, or any other name meaningful to the developer.\nOnce a `Release` refers to a `Ruleset`, the rules can be enforced by\nFirebase Rules-enabled services.\n\nMore than one `Release` may be 'live' concurrently. Consider the following\nthree `Release` names for `projects/foo` and the `Ruleset` to which they\nrefer.\n\nRelease Name                    | Ruleset Name\n--------------------------------|-------------\nprojects/foo/releases/prod      | projects/foo/rulesets/uuid123\nprojects/foo/releases/prod/beta | projects/foo/rulesets/uuid123\nprojects/foo/releases/prod/v23  | projects/foo/rulesets/uuid456\n\nThe table reflects the `Ruleset` rollout in progress. The `prod` and\n`prod/beta` releases refer to the same `Ruleset`. However, `prod/v23`\nrefers to a new `Ruleset`. The `Ruleset` reference for a `Release` may be\nupdated using the UpdateRelease method.",
	//   "flatPath": "v1/projects/{projectsId}/releases",
	//   "httpMethod": "POST",
	//   "id": "firebaserules.projects.releases.create",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name for the project which owns this `Release`.\n\nFormat: `projects/{project_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/releases",
	//   "request": {
	//     "$ref": "Release"
	//   },
	//   "response": {
	//     "$ref": "Release"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase"
	//   ]
	// }

}

// method id "firebaserules.projects.releases.delete":

type ProjectsReleasesDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Delete a `Release` by resource name.
func (r *ProjectsReleasesService) Delete(name string) *ProjectsReleasesDeleteCall {
	c := &ProjectsReleasesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReleasesDeleteCall) Fields(s ...googleapi.Field) *ProjectsReleasesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReleasesDeleteCall) Context(ctx context.Context) *ProjectsReleasesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReleasesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReleasesDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "firebaserules.projects.releases.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsReleasesDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Delete a `Release` by resource name.",
	//   "flatPath": "v1/projects/{projectsId}/releases/{releasesId}",
	//   "httpMethod": "DELETE",
	//   "id": "firebaserules.projects.releases.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name for the `Release` to delete.\n\nFormat: `projects/{project_id}/releases/{release_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/releases/.+$",
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
	//     "https://www.googleapis.com/auth/firebase"
	//   ]
	// }

}

// method id "firebaserules.projects.releases.get":

type ProjectsReleasesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get a `Release` by name.
func (r *ProjectsReleasesService) Get(name string) *ProjectsReleasesGetCall {
	c := &ProjectsReleasesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReleasesGetCall) Fields(s ...googleapi.Field) *ProjectsReleasesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsReleasesGetCall) IfNoneMatch(entityTag string) *ProjectsReleasesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReleasesGetCall) Context(ctx context.Context) *ProjectsReleasesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReleasesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReleasesGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "firebaserules.projects.releases.get" call.
// Exactly one of *Release or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Release.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsReleasesGetCall) Do(opts ...googleapi.CallOption) (*Release, error) {
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
	ret := &Release{
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
	//   "description": "Get a `Release` by name.",
	//   "flatPath": "v1/projects/{projectsId}/releases/{releasesId}",
	//   "httpMethod": "GET",
	//   "id": "firebaserules.projects.releases.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name of the `Release`.\n\nFormat: `projects/{project_id}/releases/{release_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/releases/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Release"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase",
	//     "https://www.googleapis.com/auth/firebase.readonly"
	//   ]
	// }

}

// method id "firebaserules.projects.releases.list":

type ProjectsReleasesListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List the `Release` values for a project. This list may
// optionally be
// filtered by `Release` name, `Ruleset` name, `TestSuite` name, or
// any
// combination thereof.
func (r *ProjectsReleasesService) List(name string) *ProjectsReleasesListCall {
	c := &ProjectsReleasesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": `Release` filter. The
// list method supports filters with restrictions on the
// `Release.name`, `Release.ruleset_name`, and
// `Release.test_suite_name`.
//
// Example 1: A filter of 'name=prod*' might return `Release`s with
// names
// within 'projects/foo' prefixed with 'prod':
//
// Name                          | Ruleset
// Name
// ------------------------------|-------------
// projects/foo/release
// s/prod    |
// projects/foo/rulesets/uuid1234
// projects/foo/releases/prod/v1 |
// projects/foo/rulesets/uuid1234
// projects/foo/releases/prod/v2 |
// projects/foo/rulesets/uuid8888
//
// Example 2: A filter of `name=prod* ruleset_name=uuid1234` would
// return only
// `Release` instances for 'projects/foo' with names prefixed with
// 'prod'
// referring to the same `Ruleset` name of 'uuid1234':
//
// Name                          | Ruleset
// Name
// ------------------------------|-------------
// projects/foo/release
// s/prod    | projects/foo/rulesets/1234
// projects/foo/releases/prod/v1 | projects/foo/rulesets/1234
//
// In the examples, the filter parameters refer to the search filters
// are
// relative to the project. Fully qualified prefixed may also be used.
// e.g.
// `test_suite_name=projects/foo/testsuites/uuid1`
func (c *ProjectsReleasesListCall) Filter(filter string) *ProjectsReleasesListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": Page size to load.
// Maximum of 100. Defaults to 10.
// Note: `page_size` is just a hint and the service may choose to load
// fewer
// than `page_size` results due to the size of the output. To traverse
// all of
// the releases, the caller should iterate until the `page_token` on
// the
// response is empty.
func (c *ProjectsReleasesListCall) PageSize(pageSize int64) *ProjectsReleasesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Next page token
// for the next batch of `Release` instances.
func (c *ProjectsReleasesListCall) PageToken(pageToken string) *ProjectsReleasesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReleasesListCall) Fields(s ...googleapi.Field) *ProjectsReleasesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsReleasesListCall) IfNoneMatch(entityTag string) *ProjectsReleasesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReleasesListCall) Context(ctx context.Context) *ProjectsReleasesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReleasesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReleasesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/releases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "firebaserules.projects.releases.list" call.
// Exactly one of *ListReleasesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListReleasesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsReleasesListCall) Do(opts ...googleapi.CallOption) (*ListReleasesResponse, error) {
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
	ret := &ListReleasesResponse{
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
	//   "description": "List the `Release` values for a project. This list may optionally be\nfiltered by `Release` name, `Ruleset` name, `TestSuite` name, or any\ncombination thereof.",
	//   "flatPath": "v1/projects/{projectsId}/releases",
	//   "httpMethod": "GET",
	//   "id": "firebaserules.projects.releases.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "`Release` filter. The list method supports filters with restrictions on the\n`Release.name`, `Release.ruleset_name`, and `Release.test_suite_name`.\n\nExample 1: A filter of 'name=prod*' might return `Release`s with names\nwithin 'projects/foo' prefixed with 'prod':\n\nName                          | Ruleset Name\n------------------------------|-------------\nprojects/foo/releases/prod    | projects/foo/rulesets/uuid1234\nprojects/foo/releases/prod/v1 | projects/foo/rulesets/uuid1234\nprojects/foo/releases/prod/v2 | projects/foo/rulesets/uuid8888\n\nExample 2: A filter of `name=prod* ruleset_name=uuid1234` would return only\n`Release` instances for 'projects/foo' with names prefixed with 'prod'\nreferring to the same `Ruleset` name of 'uuid1234':\n\nName                          | Ruleset Name\n------------------------------|-------------\nprojects/foo/releases/prod    | projects/foo/rulesets/1234\nprojects/foo/releases/prod/v1 | projects/foo/rulesets/1234\n\nIn the examples, the filter parameters refer to the search filters are\nrelative to the project. Fully qualified prefixed may also be used. e.g.\n`test_suite_name=projects/foo/testsuites/uuid1`",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Resource name for the project.\n\nFormat: `projects/{project_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Page size to load. Maximum of 100. Defaults to 10.\nNote: `page_size` is just a hint and the service may choose to load fewer\nthan `page_size` results due to the size of the output. To traverse all of\nthe releases, the caller should iterate until the `page_token` on the\nresponse is empty.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Next page token for the next batch of `Release` instances.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/releases",
	//   "response": {
	//     "$ref": "ListReleasesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase",
	//     "https://www.googleapis.com/auth/firebase.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsReleasesListCall) Pages(ctx context.Context, f func(*ListReleasesResponse) error) error {
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

// method id "firebaserules.projects.releases.update":

type ProjectsReleasesUpdateCall struct {
	s          *Service
	name       string
	release    *Release
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Update a `Release`.
//
// Only updates to the `ruleset_name` and `test_suite_name` fields will
// be
// honored. `Release` rename is not supported. To create a `Release` use
// the
// CreateRelease method.
func (r *ProjectsReleasesService) Update(name string, release *Release) *ProjectsReleasesUpdateCall {
	c := &ProjectsReleasesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.release = release
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReleasesUpdateCall) Fields(s ...googleapi.Field) *ProjectsReleasesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReleasesUpdateCall) Context(ctx context.Context) *ProjectsReleasesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReleasesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReleasesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.release)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "firebaserules.projects.releases.update" call.
// Exactly one of *Release or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Release.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsReleasesUpdateCall) Do(opts ...googleapi.CallOption) (*Release, error) {
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
	ret := &Release{
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
	//   "description": "Update a `Release`.\n\nOnly updates to the `ruleset_name` and `test_suite_name` fields will be\nhonored. `Release` rename is not supported. To create a `Release` use the\nCreateRelease method.",
	//   "flatPath": "v1/projects/{projectsId}/releases/{releasesId}",
	//   "httpMethod": "PUT",
	//   "id": "firebaserules.projects.releases.update",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name for the `Release`.\n\n`Release` names may be structured `app1/prod/v2` or flat `app1_prod_v2`\nwhich affords developers a great deal of flexibility in mapping the name\nto the style that best fits their existing development practices. For\nexample, a name could refer to an environment, an app, a version, or some\ncombination of three.\n\nIn the table below, for the project name `projects/foo`, the following\nrelative release paths show how flat and structured names might be chosen\nto match a desired development / deployment strategy.\n\nUse Case     | Flat Name           | Structured Name\n-------------|---------------------|----------------\nEnvironments | releases/qa         | releases/qa\nApps         | releases/app1_qa    | releases/app1/qa\nVersions     | releases/app1_v2_qa | releases/app1/v2/qa\n\nThe delimiter between the release name path elements can be almost anything\nand it should work equally well with the release name list filter, but in\nmany ways the structured paths provide a clearer picture of the\nrelationship between `Release` instances.\n\nFormat: `projects/{project_id}/releases/{release_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/releases/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "Release"
	//   },
	//   "response": {
	//     "$ref": "Release"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase"
	//   ]
	// }

}

// method id "firebaserules.projects.rulesets.create":

type ProjectsRulesetsCreateCall struct {
	s          *Service
	name       string
	ruleset    *Ruleset
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Create a `Ruleset` from `Source`.
//
// The `Ruleset` is given a unique generated name which is returned to
// the
// caller. `Source` containing syntactic or semantics errors will result
// in an
// error response indicating the first error encountered. For a detailed
// view
// of `Source` issues, use TestRuleset.
func (r *ProjectsRulesetsService) Create(name string, ruleset *Ruleset) *ProjectsRulesetsCreateCall {
	c := &ProjectsRulesetsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.ruleset = ruleset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRulesetsCreateCall) Fields(s ...googleapi.Field) *ProjectsRulesetsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRulesetsCreateCall) Context(ctx context.Context) *ProjectsRulesetsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRulesetsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRulesetsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.ruleset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/rulesets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "firebaserules.projects.rulesets.create" call.
// Exactly one of *Ruleset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Ruleset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsRulesetsCreateCall) Do(opts ...googleapi.CallOption) (*Ruleset, error) {
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
	ret := &Ruleset{
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
	//   "description": "Create a `Ruleset` from `Source`.\n\nThe `Ruleset` is given a unique generated name which is returned to the\ncaller. `Source` containing syntactic or semantics errors will result in an\nerror response indicating the first error encountered. For a detailed view\nof `Source` issues, use TestRuleset.",
	//   "flatPath": "v1/projects/{projectsId}/rulesets",
	//   "httpMethod": "POST",
	//   "id": "firebaserules.projects.rulesets.create",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name for Project which owns this `Ruleset`.\n\nFormat: `projects/{project_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/rulesets",
	//   "request": {
	//     "$ref": "Ruleset"
	//   },
	//   "response": {
	//     "$ref": "Ruleset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase"
	//   ]
	// }

}

// method id "firebaserules.projects.rulesets.delete":

type ProjectsRulesetsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Delete a `Ruleset` by resource name.
//
// If the `Ruleset` is referenced by a `Release` the operation will
// fail.
func (r *ProjectsRulesetsService) Delete(name string) *ProjectsRulesetsDeleteCall {
	c := &ProjectsRulesetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRulesetsDeleteCall) Fields(s ...googleapi.Field) *ProjectsRulesetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRulesetsDeleteCall) Context(ctx context.Context) *ProjectsRulesetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRulesetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRulesetsDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "firebaserules.projects.rulesets.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsRulesetsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Delete a `Ruleset` by resource name.\n\nIf the `Ruleset` is referenced by a `Release` the operation will fail.",
	//   "flatPath": "v1/projects/{projectsId}/rulesets/{rulesetsId}",
	//   "httpMethod": "DELETE",
	//   "id": "firebaserules.projects.rulesets.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name for the ruleset to delete.\n\nFormat: `projects/{project_id}/rulesets/{ruleset_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/rulesets/[^/]+$",
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
	//     "https://www.googleapis.com/auth/firebase"
	//   ]
	// }

}

// method id "firebaserules.projects.rulesets.get":

type ProjectsRulesetsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get a `Ruleset` by name including the full `Source` contents.
func (r *ProjectsRulesetsService) Get(name string) *ProjectsRulesetsGetCall {
	c := &ProjectsRulesetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRulesetsGetCall) Fields(s ...googleapi.Field) *ProjectsRulesetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsRulesetsGetCall) IfNoneMatch(entityTag string) *ProjectsRulesetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRulesetsGetCall) Context(ctx context.Context) *ProjectsRulesetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRulesetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRulesetsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "firebaserules.projects.rulesets.get" call.
// Exactly one of *Ruleset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Ruleset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsRulesetsGetCall) Do(opts ...googleapi.CallOption) (*Ruleset, error) {
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
	ret := &Ruleset{
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
	//   "description": "Get a `Ruleset` by name including the full `Source` contents.",
	//   "flatPath": "v1/projects/{projectsId}/rulesets/{rulesetsId}",
	//   "httpMethod": "GET",
	//   "id": "firebaserules.projects.rulesets.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name for the ruleset to get.\n\nFormat: `projects/{project_id}/rulesets/{ruleset_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/rulesets/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Ruleset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase",
	//     "https://www.googleapis.com/auth/firebase.readonly"
	//   ]
	// }

}

// method id "firebaserules.projects.rulesets.list":

type ProjectsRulesetsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List `Ruleset` metadata only and optionally filter the results
// by `Ruleset`
// name.
//
// The full `Source` contents of a `Ruleset` may be retrieved
// with
// GetRuleset.
func (r *ProjectsRulesetsService) List(name string) *ProjectsRulesetsListCall {
	c := &ProjectsRulesetsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": `Ruleset` filter. The
// list method supports filters with restrictions
// on
// `Ruleset.name`.
//
// Filters on `Ruleset.create_time` should use the `date` function
// which
// parses strings that conform to the RFC 3339 date/time
// specifications.
//
// Example: `create_time > date("2017-01-01") AND name=UUID-*`
func (c *ProjectsRulesetsListCall) Filter(filter string) *ProjectsRulesetsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": Page size to load.
// Maximum of 100. Defaults to 10.
// Note: `page_size` is just a hint and the service may choose to load
// less
// than `page_size` due to the size of the output. To traverse all of
// the
// releases, caller should iterate until the `page_token` is empty.
func (c *ProjectsRulesetsListCall) PageSize(pageSize int64) *ProjectsRulesetsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Next page token
// for loading the next batch of `Ruleset` instances.
func (c *ProjectsRulesetsListCall) PageToken(pageToken string) *ProjectsRulesetsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRulesetsListCall) Fields(s ...googleapi.Field) *ProjectsRulesetsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsRulesetsListCall) IfNoneMatch(entityTag string) *ProjectsRulesetsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRulesetsListCall) Context(ctx context.Context) *ProjectsRulesetsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRulesetsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRulesetsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/rulesets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "firebaserules.projects.rulesets.list" call.
// Exactly one of *ListRulesetsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListRulesetsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsRulesetsListCall) Do(opts ...googleapi.CallOption) (*ListRulesetsResponse, error) {
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
	ret := &ListRulesetsResponse{
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
	//   "description": "List `Ruleset` metadata only and optionally filter the results by `Ruleset`\nname.\n\nThe full `Source` contents of a `Ruleset` may be retrieved with\nGetRuleset.",
	//   "flatPath": "v1/projects/{projectsId}/rulesets",
	//   "httpMethod": "GET",
	//   "id": "firebaserules.projects.rulesets.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "`Ruleset` filter. The list method supports filters with restrictions on\n`Ruleset.name`.\n\nFilters on `Ruleset.create_time` should use the `date` function which\nparses strings that conform to the RFC 3339 date/time specifications.\n\nExample: `create_time \u003e date(\"2017-01-01\") AND name=UUID-*`",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Resource name for the project.\n\nFormat: `projects/{project_id}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Page size to load. Maximum of 100. Defaults to 10.\nNote: `page_size` is just a hint and the service may choose to load less\nthan `page_size` due to the size of the output. To traverse all of the\nreleases, caller should iterate until the `page_token` is empty.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Next page token for loading the next batch of `Ruleset` instances.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/rulesets",
	//   "response": {
	//     "$ref": "ListRulesetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/firebase",
	//     "https://www.googleapis.com/auth/firebase.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsRulesetsListCall) Pages(ctx context.Context, f func(*ListRulesetsResponse) error) error {
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
