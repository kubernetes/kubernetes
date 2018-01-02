// Package servicemanagement provides access to the Google Service Management API.
//
// See https://cloud.google.com/service-management/
//
// Usage example:
//
//   import "google.golang.org/api/servicemanagement/v1"
//   ...
//   servicemanagementService, err := servicemanagement.New(oauthHttpClient)
package servicemanagement // import "google.golang.org/api/servicemanagement/v1"

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

const apiId = "servicemanagement:v1"
const apiName = "servicemanagement"
const apiVersion = "v1"
const basePath = "https://servicemanagement.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// Manage your Google API service configuration
	ServiceManagementScope = "https://www.googleapis.com/auth/service.management"

	// View your Google API service configuration
	ServiceManagementReadonlyScope = "https://www.googleapis.com/auth/service.management.readonly"
)

func New(client *http.Client) (*APIService, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &APIService{client: client, BasePath: basePath}
	s.Operations = NewOperationsService(s)
	s.Services = NewServicesService(s)
	return s, nil
}

type APIService struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Operations *OperationsService

	Services *ServicesService
}

func (s *APIService) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewOperationsService(s *APIService) *OperationsService {
	rs := &OperationsService{s: s}
	return rs
}

type OperationsService struct {
	s *APIService
}

func NewServicesService(s *APIService) *ServicesService {
	rs := &ServicesService{s: s}
	rs.Configs = NewServicesConfigsService(s)
	rs.Consumers = NewServicesConsumersService(s)
	rs.Rollouts = NewServicesRolloutsService(s)
	return rs
}

type ServicesService struct {
	s *APIService

	Configs *ServicesConfigsService

	Consumers *ServicesConsumersService

	Rollouts *ServicesRolloutsService
}

func NewServicesConfigsService(s *APIService) *ServicesConfigsService {
	rs := &ServicesConfigsService{s: s}
	return rs
}

type ServicesConfigsService struct {
	s *APIService
}

func NewServicesConsumersService(s *APIService) *ServicesConsumersService {
	rs := &ServicesConsumersService{s: s}
	return rs
}

type ServicesConsumersService struct {
	s *APIService
}

func NewServicesRolloutsService(s *APIService) *ServicesRolloutsService {
	rs := &ServicesRolloutsService{s: s}
	return rs
}

type ServicesRolloutsService struct {
	s *APIService
}

// Advice: Generated advice about this change, used for providing
// more
// information about how a change will affect the existing service.
type Advice struct {
	// Description: Useful description for why this advice was applied and
	// what actions should
	// be taken to mitigate any implied risks.
	Description string `json:"description,omitempty"`

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

func (s *Advice) MarshalJSON() ([]byte, error) {
	type noMethod Advice
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Api: Api is a light-weight descriptor for an API
// Interface.
//
// Interfaces are also described as "protocol buffer services" in some
// contexts,
// such as by the "service" keyword in a .proto file, but they are
// different
// from API Services, which represent a concrete implementation of an
// interface
// as opposed to simply a description of methods and bindings. They are
// also
// sometimes simply referred to as "APIs" in other contexts, such as the
// name of
// this message itself. See
// https://cloud.google.com/apis/design/glossary for
// detailed terminology.
type Api struct {
	// Methods: The methods of this interface, in unspecified order.
	Methods []*Method `json:"methods,omitempty"`

	// Mixins: Included interfaces. See Mixin.
	Mixins []*Mixin `json:"mixins,omitempty"`

	// Name: The fully qualified name of this interface, including package
	// name
	// followed by the interface's simple name.
	Name string `json:"name,omitempty"`

	// Options: Any metadata attached to the interface.
	Options []*Option `json:"options,omitempty"`

	// SourceContext: Source context for the protocol buffer service
	// represented by this
	// message.
	SourceContext *SourceContext `json:"sourceContext,omitempty"`

	// Syntax: The source syntax of the service.
	//
	// Possible values:
	//   "SYNTAX_PROTO2" - Syntax `proto2`.
	//   "SYNTAX_PROTO3" - Syntax `proto3`.
	Syntax string `json:"syntax,omitempty"`

	// Version: A version string for this interface. If specified, must have
	// the form
	// `major-version.minor-version`, as in `1.10`. If the minor version
	// is
	// omitted, it defaults to zero. If the entire version field is empty,
	// the
	// major version is derived from the package name, as outlined below. If
	// the
	// field is not empty, the version in the package name will be verified
	// to be
	// consistent with what is provided here.
	//
	// The versioning schema uses [semantic
	// versioning](http://semver.org) where the major version
	// number
	// indicates a breaking change and the minor version an
	// additive,
	// non-breaking change. Both version numbers are signals to users
	// what to expect from different versions, and should be
	// carefully
	// chosen based on the product plan.
	//
	// The major version is also reflected in the package name of
	// the
	// interface, which must end in `v<major-version>`, as
	// in
	// `google.feature.v1`. For major versions 0 and 1, the suffix can
	// be omitted. Zero major versions must only be used for
	// experimental, non-GA interfaces.
	//
	Version string `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Methods") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Methods") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Api) MarshalJSON() ([]byte, error) {
	type noMethod Api
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuditConfig: Specifies the audit configuration for a service.
// The configuration determines which permission types are logged, and
// what
// identities, if any, are exempted from logging.
// An AuditConfig must have one or more AuditLogConfigs.
//
// If there are AuditConfigs for both `allServices` and a specific
// service,
// the union of the two AuditConfigs is used for that service: the
// log_types
// specified in each AuditConfig are enabled, and the exempted_members
// in each
// AuditConfig are exempted.
//
// Example Policy with multiple AuditConfigs:
//
//     {
//       "audit_configs": [
//         {
//           "service": "allServices"
//           "audit_log_configs": [
//             {
//               "log_type": "DATA_READ",
//               "exempted_members": [
//                 "user:foo@gmail.com"
//               ]
//             },
//             {
//               "log_type": "DATA_WRITE",
//             },
//             {
//               "log_type": "ADMIN_READ",
//             }
//           ]
//         },
//         {
//           "service": "fooservice.googleapis.com"
//           "audit_log_configs": [
//             {
//               "log_type": "DATA_READ",
//             },
//             {
//               "log_type": "DATA_WRITE",
//               "exempted_members": [
//                 "user:bar@gmail.com"
//               ]
//             }
//           ]
//         }
//       ]
//     }
//
// For fooservice, this policy enables DATA_READ, DATA_WRITE and
// ADMIN_READ
// logging. It also exempts foo@gmail.com from DATA_READ logging,
// and
// bar@gmail.com from DATA_WRITE logging.
type AuditConfig struct {
	// AuditLogConfigs: The configuration for logging of each type of
	// permission.
	// Next ID: 4
	AuditLogConfigs []*AuditLogConfig `json:"auditLogConfigs,omitempty"`

	ExemptedMembers []string `json:"exemptedMembers,omitempty"`

	// Service: Specifies a service that will be enabled for audit
	// logging.
	// For example, `storage.googleapis.com`,
	// `cloudsql.googleapis.com`.
	// `allServices` is a special value that covers all services.
	Service string `json:"service,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuditLogConfigs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuditLogConfigs") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AuditConfig) MarshalJSON() ([]byte, error) {
	type noMethod AuditConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuditLogConfig: Provides the configuration for logging a type of
// permissions.
// Example:
//
//     {
//       "audit_log_configs": [
//         {
//           "log_type": "DATA_READ",
//           "exempted_members": [
//             "user:foo@gmail.com"
//           ]
//         },
//         {
//           "log_type": "DATA_WRITE",
//         }
//       ]
//     }
//
// This enables 'DATA_READ' and 'DATA_WRITE' logging, while
// exempting
// foo@gmail.com from DATA_READ logging.
type AuditLogConfig struct {
	// ExemptedMembers: Specifies the identities that do not cause logging
	// for this type of
	// permission.
	// Follows the same format of Binding.members.
	ExemptedMembers []string `json:"exemptedMembers,omitempty"`

	// LogType: The log type that this config enables.
	//
	// Possible values:
	//   "LOG_TYPE_UNSPECIFIED" - Default case. Should never be this.
	//   "ADMIN_READ" - Admin reads. Example: CloudIAM getIamPolicy
	//   "DATA_WRITE" - Data writes. Example: CloudSQL Users create
	//   "DATA_READ" - Data reads. Example: CloudSQL Users list
	LogType string `json:"logType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExemptedMembers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExemptedMembers") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AuditLogConfig) MarshalJSON() ([]byte, error) {
	type noMethod AuditLogConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthProvider: Configuration for an anthentication provider, including
// support for
// [JSON Web Token
// (JWT)](https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32)
// .
type AuthProvider struct {
	// Audiences: The list of
	// JWT
	// [audiences](https://tools.ietf.org/html/draft-ietf-oauth-json-web-
	// token-32#section-4.1.3).
	// that are allowed to access. A JWT containing any of these audiences
	// will
	// be accepted. When this setting is absent, only JWTs with
	// audience
	// "https://Service_name/API_name"
	// will be accepted. For example, if no audiences are in the
	// setting,
	// LibraryService API will only accept JWTs with the following
	// audience
	// "https://library-example.googleapis.com/google.example.librar
	// y.v1.LibraryService".
	//
	// Example:
	//
	//     audiences: bookstore_android.apps.googleusercontent.com,
	//                bookstore_web.apps.googleusercontent.com
	Audiences string `json:"audiences,omitempty"`

	// AuthorizationUrl: Redirect URL if JWT token is required but no
	// present or is expired.
	// Implement authorizationUrl of securityDefinitions in OpenAPI spec.
	AuthorizationUrl string `json:"authorizationUrl,omitempty"`

	// Id: The unique identifier of the auth provider. It will be referred
	// to by
	// `AuthRequirement.provider_id`.
	//
	// Example: "bookstore_auth".
	Id string `json:"id,omitempty"`

	// Issuer: Identifies the principal that issued the JWT.
	// See
	// https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32#sec
	// tion-4.1.1
	// Usually a URL or an email address.
	//
	// Example: https://securetoken.google.com
	// Example: 1234567-compute@developer.gserviceaccount.com
	Issuer string `json:"issuer,omitempty"`

	// JwksUri: URL of the provider's public key set to validate signature
	// of the JWT. See
	// [OpenID
	// Discovery](https://openid.net/specs/openid-connect-discovery-1_0.html#
	// ProviderMetadata).
	// Optional if the key set document:
	//  - can be retrieved from
	//    [OpenID
	// Discovery](https://openid.net/specs/openid-connect-discovery-1_0.html
	//
	//    of the issuer.
	//  - can be inferred from the email domain of the issuer (e.g. a Google
	// service account).
	//
	// Example: https://www.googleapis.com/oauth2/v1/certs
	JwksUri string `json:"jwksUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Audiences") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Audiences") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuthProvider) MarshalJSON() ([]byte, error) {
	type noMethod AuthProvider
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthRequirement: User-defined authentication requirements, including
// support for
// [JSON Web Token
// (JWT)](https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32)
// .
type AuthRequirement struct {
	// Audiences: NOTE: This will be deprecated soon, once
	// AuthProvider.audiences is
	// implemented and accepted in all the runtime components.
	//
	// The list of
	// JWT
	// [audiences](https://tools.ietf.org/html/draft-ietf-oauth-json-web-
	// token-32#section-4.1.3).
	// that are allowed to access. A JWT containing any of these audiences
	// will
	// be accepted. When this setting is absent, only JWTs with
	// audience
	// "https://Service_name/API_name"
	// will be accepted. For example, if no audiences are in the
	// setting,
	// LibraryService API will only accept JWTs with the following
	// audience
	// "https://library-example.googleapis.com/google.example.librar
	// y.v1.LibraryService".
	//
	// Example:
	//
	//     audiences: bookstore_android.apps.googleusercontent.com,
	//                bookstore_web.apps.googleusercontent.com
	Audiences string `json:"audiences,omitempty"`

	// ProviderId: id from authentication provider.
	//
	// Example:
	//
	//     provider_id: bookstore_auth
	ProviderId string `json:"providerId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Audiences") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Audiences") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuthRequirement) MarshalJSON() ([]byte, error) {
	type noMethod AuthRequirement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Authentication: `Authentication` defines the authentication
// configuration for an API.
//
// Example for an API targeted for external use:
//
//     name: calendar.googleapis.com
//     authentication:
//       providers:
//       - id: google_calendar_auth
//         jwks_uri: https://www.googleapis.com/oauth2/v1/certs
//         issuer: https://securetoken.google.com
//       rules:
//       - selector: "*"
//         requirements:
//           provider_id: google_calendar_auth
type Authentication struct {
	// Providers: Defines a set of authentication providers that a service
	// supports.
	Providers []*AuthProvider `json:"providers,omitempty"`

	// Rules: A list of authentication rules that apply to individual API
	// methods.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*AuthenticationRule `json:"rules,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Providers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Providers") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Authentication) MarshalJSON() ([]byte, error) {
	type noMethod Authentication
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthenticationRule: Authentication rules for the service.
//
// By default, if a method has any authentication requirements, every
// request
// must include a valid credential matching one of the
// requirements.
// It's an error to include more than one kind of credential in a
// single
// request.
//
// If a method doesn't have any auth requirements, request credentials
// will be
// ignored.
type AuthenticationRule struct {
	// AllowWithoutCredential: Whether to allow requests without a
	// credential. The credential can be
	// an OAuth token, Google cookies (first-party auth) or
	// EndUserCreds.
	//
	// For requests without credentials, if the service control environment
	// is
	// specified, each incoming request **must** be associated with a
	// service
	// consumer. This can be done by passing an API key that belongs to a
	// consumer
	// project.
	AllowWithoutCredential bool `json:"allowWithoutCredential,omitempty"`

	// CustomAuth: Configuration for custom authentication.
	CustomAuth *CustomAuthRequirements `json:"customAuth,omitempty"`

	// Oauth: The requirements for OAuth credentials.
	Oauth *OAuthRequirements `json:"oauth,omitempty"`

	// Requirements: Requirements for additional authentication providers.
	Requirements []*AuthRequirement `json:"requirements,omitempty"`

	// Selector: Selects the methods to which this rule applies.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AllowWithoutCredential") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowWithoutCredential")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AuthenticationRule) MarshalJSON() ([]byte, error) {
	type noMethod AuthenticationRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthorizationConfig: Configuration of authorization.
//
// This section determines the authorization provider, if unspecified,
// then no
// authorization check will be done.
//
// Example:
//
//     experimental:
//       authorization:
//         provider: firebaserules.googleapis.com
type AuthorizationConfig struct {
	// Provider: The name of the authorization provider, such
	// as
	// firebaserules.googleapis.com.
	Provider string `json:"provider,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Provider") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Provider") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuthorizationConfig) MarshalJSON() ([]byte, error) {
	type noMethod AuthorizationConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Backend: `Backend` defines the backend configuration for a service.
type Backend struct {
	// Rules: A list of API backend rules that apply to individual API
	// methods.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*BackendRule `json:"rules,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Rules") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Rules") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Backend) MarshalJSON() ([]byte, error) {
	type noMethod Backend
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BackendRule: A backend rule provides configuration for an individual
// API element.
type BackendRule struct {
	// Address: The address of the API backend.
	Address string `json:"address,omitempty"`

	// Deadline: The number of seconds to wait for a response from a
	// request.  The default
	// deadline for gRPC and HTTP requests is 5 seconds.
	Deadline float64 `json:"deadline,omitempty"`

	// MinDeadline: Minimum deadline in seconds needed for this method.
	// Calls having deadline
	// value lower than this will be rejected.
	MinDeadline float64 `json:"minDeadline,omitempty"`

	// Selector: Selects the methods to which this rule applies.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Address") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Address") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BackendRule) MarshalJSON() ([]byte, error) {
	type noMethod BackendRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *BackendRule) UnmarshalJSON(data []byte) error {
	type noMethod BackendRule
	var s1 struct {
		Deadline    gensupport.JSONFloat64 `json:"deadline"`
		MinDeadline gensupport.JSONFloat64 `json:"minDeadline"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Deadline = float64(s1.Deadline)
	s.MinDeadline = float64(s1.MinDeadline)
	return nil
}

// Binding: Associates `members` with a `role`.
type Binding struct {
	// Condition: The condition that is associated with this binding.
	// NOTE: an unsatisfied condition will not allow user access via
	// current
	// binding. Different bindings, including their conditions, are
	// examined
	// independently.
	// This field is GOOGLE_INTERNAL.
	Condition *Expr `json:"condition,omitempty"`

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

func (s *Binding) MarshalJSON() ([]byte, error) {
	type noMethod Binding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ChangeReport: Change report associated with a particular service
// configuration.
//
// It contains a list of ConfigChanges based on the comparison
// between
// two service configurations.
type ChangeReport struct {
	// ConfigChanges: List of changes between two service
	// configurations.
	// The changes will be alphabetically sorted based on the identifier
	// of each change.
	// A ConfigChange identifier is a dot separated path to the
	// configuration.
	// Example:
	// visibility.rules[selector='LibraryService.CreateBook'].restriction
	ConfigChanges []*ConfigChange `json:"configChanges,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConfigChanges") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConfigChanges") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ChangeReport) MarshalJSON() ([]byte, error) {
	type noMethod ChangeReport
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ConfigChange: Output generated from semantically comparing two
// versions of a service
// configuration.
//
// Includes detailed information about a field that have changed
// with
// applicable advice about potential consequences for the change, such
// as
// backwards-incompatibility.
type ConfigChange struct {
	// Advices: Collection of advice provided for this change, useful for
	// determining the
	// possible impact of this change.
	Advices []*Advice `json:"advices,omitempty"`

	// ChangeType: The type for this change, either ADDED, REMOVED, or
	// MODIFIED.
	//
	// Possible values:
	//   "CHANGE_TYPE_UNSPECIFIED" - No value was provided.
	//   "ADDED" - The changed object exists in the 'new' service
	// configuration, but not
	// in the 'old' service configuration.
	//   "REMOVED" - The changed object exists in the 'old' service
	// configuration, but not
	// in the 'new' service configuration.
	//   "MODIFIED" - The changed object exists in both service
	// configurations, but its value
	// is different.
	ChangeType string `json:"changeType,omitempty"`

	// Element: Object hierarchy path to the change, with levels separated
	// by a '.'
	// character. For repeated fields, an applicable unique identifier field
	// is
	// used for the index (usually selector, name, or id). For maps, the
	// term
	// 'key' is used. If the field has no unique identifier, the numeric
	// index
	// is used.
	// Examples:
	// -
	// visibility.rules[selector=="google.LibraryService.CreateBook"].restric
	// tion
	// -
	// quota.metric_rules[selector=="google"].metric_costs[key=="reads"].valu
	// e
	// - logging.producer_destinations[0]
	Element string `json:"element,omitempty"`

	// NewValue: Value of the changed object in the new Service
	// configuration,
	// in JSON format. This field will not be populated if ChangeType ==
	// REMOVED.
	NewValue string `json:"newValue,omitempty"`

	// OldValue: Value of the changed object in the old Service
	// configuration,
	// in JSON format. This field will not be populated if ChangeType ==
	// ADDED.
	OldValue string `json:"oldValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Advices") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Advices") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ConfigChange) MarshalJSON() ([]byte, error) {
	type noMethod ConfigChange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ConfigFile: Generic specification of a source configuration file
type ConfigFile struct {
	// FileContents: The bytes that constitute the file.
	FileContents string `json:"fileContents,omitempty"`

	// FilePath: The file name of the configuration file (full or relative
	// path).
	FilePath string `json:"filePath,omitempty"`

	// FileType: The type of configuration file this represents.
	//
	// Possible values:
	//   "FILE_TYPE_UNSPECIFIED" - Unknown file type.
	//   "SERVICE_CONFIG_YAML" - YAML-specification of service.
	//   "OPEN_API_JSON" - OpenAPI specification, serialized in JSON.
	//   "OPEN_API_YAML" - OpenAPI specification, serialized in YAML.
	//   "FILE_DESCRIPTOR_SET_PROTO" - FileDescriptorSet, generated by
	// protoc.
	//
	// To generate, use protoc with imports and source info included.
	// For an example test.proto file, the following command would put the
	// value
	// in a new file named out.pb.
	//
	// $protoc --include_imports --include_source_info test.proto -o out.pb
	//   "PROTO_FILE" - Uncompiled Proto file. Used for storage and display
	// purposes only,
	// currently server-side compilation is not supported. Should match
	// the
	// inputs to 'protoc' command used to generated
	// FILE_DESCRIPTOR_SET_PROTO. A
	// file of this type can only be included if at least one file of
	// type
	// FILE_DESCRIPTOR_SET_PROTO is included.
	FileType string `json:"fileType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FileContents") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FileContents") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ConfigFile) MarshalJSON() ([]byte, error) {
	type noMethod ConfigFile
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ConfigRef: Represents a service configuration with its name and id.
type ConfigRef struct {
	// Name: Resource name of a service config. It must have the
	// following
	// format: "services/{service name}/configs/{config id}".
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

func (s *ConfigRef) MarshalJSON() ([]byte, error) {
	type noMethod ConfigRef
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ConfigSource: Represents a source file which is used to generate the
// service configuration
// defined by `google.api.Service`.
type ConfigSource struct {
	// Files: Set of source configuration files that are used to generate a
	// service
	// configuration (`google.api.Service`).
	Files []*ConfigFile `json:"files,omitempty"`

	// Id: A unique ID for a specific instance of this message, typically
	// assigned
	// by the client for tracking purpose. If empty, the server may choose
	// to
	// generate one instead.
	Id string `json:"id,omitempty"`

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

func (s *ConfigSource) MarshalJSON() ([]byte, error) {
	type noMethod ConfigSource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Context: `Context` defines which contexts an API
// requests.
//
// Example:
//
//     context:
//       rules:
//       - selector: "*"
//         requested:
//         - google.rpc.context.ProjectContext
//         - google.rpc.context.OriginContext
//
// The above specifies that all methods in the API
// request
// `google.rpc.context.ProjectContext`
// and
// `google.rpc.context.OriginContext`.
//
// Available context types are defined in package
// `google.rpc.context`.
type Context struct {
	// Rules: A list of RPC context rules that apply to individual API
	// methods.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*ContextRule `json:"rules,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Rules") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Rules") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Context) MarshalJSON() ([]byte, error) {
	type noMethod Context
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContextRule: A context rule provides information about the context
// for an individual API
// element.
type ContextRule struct {
	// Provided: A list of full type names of provided contexts.
	Provided []string `json:"provided,omitempty"`

	// Requested: A list of full type names of requested contexts.
	Requested []string `json:"requested,omitempty"`

	// Selector: Selects the methods to which this rule applies.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Provided") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Provided") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ContextRule) MarshalJSON() ([]byte, error) {
	type noMethod ContextRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Control: Selects and configures the service controller used by the
// service.  The
// service controller handles features like abuse, quota, billing,
// logging,
// monitoring, etc.
type Control struct {
	// Environment: The service control environment to use. If empty, no
	// control plane
	// feature (like quota and billing) will be enabled.
	Environment string `json:"environment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Environment") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Environment") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Control) MarshalJSON() ([]byte, error) {
	type noMethod Control
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CustomAuthRequirements: Configuration for a custom authentication
// provider.
type CustomAuthRequirements struct {
	// Provider: A configuration string containing connection information
	// for the
	// authentication provider, typically formatted as a SmartService
	// string
	// (go/smartservice).
	Provider string `json:"provider,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Provider") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Provider") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CustomAuthRequirements) MarshalJSON() ([]byte, error) {
	type noMethod CustomAuthRequirements
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CustomError: Customize service error responses.  For example, list
// any service
// specific protobuf types that can appear in error detail lists
// of
// error responses.
//
// Example:
//
//     custom_error:
//       types:
//       - google.foo.v1.CustomError
//       - google.foo.v1.AnotherError
type CustomError struct {
	// Rules: The list of custom error rules that apply to individual API
	// messages.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*CustomErrorRule `json:"rules,omitempty"`

	// Types: The list of custom error detail types, e.g.
	// 'google.foo.v1.CustomError'.
	Types []string `json:"types,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Rules") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Rules") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CustomError) MarshalJSON() ([]byte, error) {
	type noMethod CustomError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CustomErrorRule: A custom error rule.
type CustomErrorRule struct {
	// IsErrorType: Mark this message as possible payload in error response.
	//  Otherwise,
	// objects of this type will be filtered when they appear in error
	// payload.
	IsErrorType bool `json:"isErrorType,omitempty"`

	// Selector: Selects messages to which this rule applies.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IsErrorType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "IsErrorType") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CustomErrorRule) MarshalJSON() ([]byte, error) {
	type noMethod CustomErrorRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CustomHttpPattern: A custom pattern is used for defining custom HTTP
// verb.
type CustomHttpPattern struct {
	// Kind: The name of this custom HTTP verb.
	Kind string `json:"kind,omitempty"`

	// Path: The path matched by this custom verb.
	Path string `json:"path,omitempty"`

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

func (s *CustomHttpPattern) MarshalJSON() ([]byte, error) {
	type noMethod CustomHttpPattern
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteServiceStrategy: Strategy used to delete a service. This
// strategy is a placeholder only
// used by the system generated rollout to delete a service.
type DeleteServiceStrategy struct {
}

// Diagnostic: Represents a diagnostic message (error or warning)
type Diagnostic struct {
	// Kind: The kind of diagnostic information provided.
	//
	// Possible values:
	//   "WARNING" - Warnings and errors
	//   "ERROR" - Only errors
	Kind string `json:"kind,omitempty"`

	// Location: File name and line number of the error or warning.
	Location string `json:"location,omitempty"`

	// Message: Message describing the error or warning.
	Message string `json:"message,omitempty"`

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

func (s *Diagnostic) MarshalJSON() ([]byte, error) {
	type noMethod Diagnostic
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DisableServiceRequest: Request message for DisableService method.
type DisableServiceRequest struct {
	// ConsumerId: The identity of consumer resource which service
	// disablement will be
	// applied to.
	//
	// The Google Service Management implementation accepts the
	// following
	// forms:
	// - "project:<project_id>"
	//
	// Note: this is made compatible
	// with
	// google.api.servicecontrol.v1.Operation.consumer_id.
	ConsumerId string `json:"consumerId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConsumerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConsumerId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DisableServiceRequest) MarshalJSON() ([]byte, error) {
	type noMethod DisableServiceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Documentation: `Documentation` provides the information for
// describing a service.
//
// Example:
// <pre><code>documentation:
//   summary: >
//     The Google Calendar API gives access
//     to most calendar features.
//   pages:
//   - name: Overview
//     content: &#40;== include google/foo/overview.md ==&#41;
//   - name: Tutorial
//     content: &#40;== include google/foo/tutorial.md ==&#41;
//     subpages;
//     - name: Java
//       content: &#40;== include google/foo/tutorial_java.md ==&#41;
//   rules:
//   - selector: google.calendar.Calendar.Get
//     description: >
//       ...
//   - selector: google.calendar.Calendar.Put
//     description: >
//       ...
// </code></pre>
// Documentation is provided in markdown syntax. In addition to
// standard markdown features, definition lists, tables and fenced
// code blocks are supported. Section headers can be provided and
// are
// interpreted relative to the section nesting of the context where
// a documentation fragment is embedded.
//
// Documentation from the IDL is merged with documentation defined
// via the config at normalization time, where documentation provided
// by config rules overrides IDL provided.
//
// A number of constructs specific to the API platform are supported
// in documentation text.
//
// In order to reference a proto element, the following
// notation can be
// used:
// <pre><code>&#91;fully.qualified.proto.name]&#91;]</code></pre>
// T
// o override the display text used for the link, this can be
// used:
// <pre><code>&#91;display
// text]&#91;fully.qualified.proto.name]</code></pre>
// Text can be excluded from doc using the following
// notation:
// <pre><code>&#40;-- internal comment --&#41;</code></pre>
// Comments can be made conditional using a visibility label. The
// below
// text will be only rendered if the `BETA` label is
// available:
// <pre><code>&#40;--BETA: comment for BETA users --&#41;</code></pre>
// A few directives are available in documentation. Note that
// directives must appear on a single line to be properly
// identified. The `include` directive includes a markdown file from
// an external source:
// <pre><code>&#40;== include path/to/file ==&#41;</code></pre>
// The `resource_for` directive marks a message to be the resource of
// a collection in REST view. If it is not specified, tools attempt
// to infer the resource from the operations in a
// collection:
// <pre><code>&#40;== resource_for v1.shelves.books
// ==&#41;</code></pre>
// The directive `suppress_warning` does not directly affect
// documentation
// and is documented together with service config validation.
type Documentation struct {
	// DocumentationRootUrl: The URL to the root of documentation.
	DocumentationRootUrl string `json:"documentationRootUrl,omitempty"`

	// Overview: Declares a single overview page. For
	// example:
	// <pre><code>documentation:
	//   summary: ...
	//   overview: &#40;== include overview.md ==&#41;
	// </code></pre>
	// This is a shortcut for the following declaration (using pages
	// style):
	// <pre><code>documentation:
	//   summary: ...
	//   pages:
	//   - name: Overview
	//     content: &#40;== include overview.md ==&#41;
	// </code></pre>
	// Note: you cannot specify both `overview` field and `pages` field.
	Overview string `json:"overview,omitempty"`

	// Pages: The top level pages for the documentation set.
	Pages []*Page `json:"pages,omitempty"`

	// Rules: A list of documentation rules that apply to individual API
	// elements.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*DocumentationRule `json:"rules,omitempty"`

	// Summary: A short summary of what the service does. Can only be
	// provided by
	// plain text.
	Summary string `json:"summary,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DocumentationRootUrl") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DocumentationRootUrl") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Documentation) MarshalJSON() ([]byte, error) {
	type noMethod Documentation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DocumentationRule: A documentation rule provides information about
// individual API elements.
type DocumentationRule struct {
	// DeprecationDescription: Deprecation description of the selected
	// element(s). It can be provided if an
	// element is marked as `deprecated`.
	DeprecationDescription string `json:"deprecationDescription,omitempty"`

	// Description: Description of the selected API(s).
	Description string `json:"description,omitempty"`

	// Selector: The selector is a comma-separated list of patterns. Each
	// pattern is a
	// qualified name of the element which may end in "*", indicating a
	// wildcard.
	// Wildcards are only allowed at the end and for a whole component of
	// the
	// qualified name, i.e. "foo.*" is ok, but not "foo.b*" or "foo.*.bar".
	// To
	// specify a default for all applicable elements, the whole pattern
	// "*"
	// is used.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DeprecationDescription") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeprecationDescription")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DocumentationRule) MarshalJSON() ([]byte, error) {
	type noMethod DocumentationRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EnableServiceRequest: Request message for EnableService method.
type EnableServiceRequest struct {
	// ConsumerId: The identity of consumer resource which service
	// enablement will be
	// applied to.
	//
	// The Google Service Management implementation accepts the
	// following
	// forms:
	// - "project:<project_id>"
	//
	// Note: this is made compatible
	// with
	// google.api.servicecontrol.v1.Operation.consumer_id.
	ConsumerId string `json:"consumerId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConsumerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConsumerId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EnableServiceRequest) MarshalJSON() ([]byte, error) {
	type noMethod EnableServiceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Endpoint: `Endpoint` describes a network endpoint that serves a set
// of APIs.
// A service may expose any number of endpoints, and all endpoints share
// the
// same service configuration, such as quota configuration and
// monitoring
// configuration.
//
// Example service configuration:
//
//     name: library-example.googleapis.com
//     endpoints:
//       # Below entry makes 'google.example.library.v1.Library'
//       # API be served from endpoint address
// library-example.googleapis.com.
//       # It also allows HTTP OPTIONS calls to be passed to the
// backend, for
//       # it to decide whether the subsequent cross-origin request is
//       # allowed to proceed.
//     - name: library-example.googleapis.com
//       allow_cors: true
type Endpoint struct {
	// Aliases: DEPRECATED: This field is no longer supported. Instead of
	// using aliases,
	// please specify multiple google.api.Endpoint for each of the
	// intented
	// alias.
	//
	// Additional names that this endpoint will be hosted on.
	Aliases []string `json:"aliases,omitempty"`

	// AllowCors:
	// Allowing
	// [CORS](https://en.wikipedia.org/wiki/Cross-origin_resource_sh
	// aring), aka
	// cross-domain traffic, would allow the backends served from this
	// endpoint to
	// receive and respond to HTTP OPTIONS requests. The response will be
	// used by
	// the browser to determine whether the subsequent cross-origin request
	// is
	// allowed to proceed.
	AllowCors bool `json:"allowCors,omitempty"`

	// Apis: The list of APIs served by this endpoint.
	//
	// If no APIs are specified this translates to "all APIs" exported by
	// the
	// service, as defined in the top-level service configuration.
	Apis []string `json:"apis,omitempty"`

	// Features: The list of features enabled on this endpoint.
	Features []string `json:"features,omitempty"`

	// Name: The canonical name of this endpoint.
	Name string `json:"name,omitempty"`

	// Target: The specification of an Internet routable address of API
	// frontend that will
	// handle requests to this [API
	// Endpoint](https://cloud.google.com/apis/design/glossary).
	// It should be either a valid IPv4 address or a fully-qualified domain
	// name.
	// For example, "8.8.8.8" or "myservice.appspot.com".
	Target string `json:"target,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Aliases") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Aliases") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Endpoint) MarshalJSON() ([]byte, error) {
	type noMethod Endpoint
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Enum: Enum type definition.
type Enum struct {
	// Enumvalue: Enum value definitions.
	Enumvalue []*EnumValue `json:"enumvalue,omitempty"`

	// Name: Enum type name.
	Name string `json:"name,omitempty"`

	// Options: Protocol buffer options.
	Options []*Option `json:"options,omitempty"`

	// SourceContext: The source context.
	SourceContext *SourceContext `json:"sourceContext,omitempty"`

	// Syntax: The source syntax.
	//
	// Possible values:
	//   "SYNTAX_PROTO2" - Syntax `proto2`.
	//   "SYNTAX_PROTO3" - Syntax `proto3`.
	Syntax string `json:"syntax,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Enumvalue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Enumvalue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Enum) MarshalJSON() ([]byte, error) {
	type noMethod Enum
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EnumValue: Enum value definition.
type EnumValue struct {
	// Name: Enum value name.
	Name string `json:"name,omitempty"`

	// Number: Enum value number.
	Number int64 `json:"number,omitempty"`

	// Options: Protocol buffer options.
	Options []*Option `json:"options,omitempty"`

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

func (s *EnumValue) MarshalJSON() ([]byte, error) {
	type noMethod EnumValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Experimental: Experimental service configuration. These configuration
// options can
// only be used by whitelisted users.
type Experimental struct {
	// Authorization: Authorization configuration.
	Authorization *AuthorizationConfig `json:"authorization,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Authorization") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Authorization") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Experimental) MarshalJSON() ([]byte, error) {
	type noMethod Experimental
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Expr: Represents an expression text. Example:
//
//     title: "User account presence"
//     description: "Determines whether the request has a user account"
//     expression: "size(request.user) > 0"
type Expr struct {
	// Description: An optional description of the expression. This is a
	// longer text which
	// describes the expression, e.g. when hovered over it in a UI.
	Description string `json:"description,omitempty"`

	// Expression: Textual representation of an expression in
	// Common Expression Language syntax.
	//
	// The application context of the containing message determines
	// which
	// well-known feature set of CEL is supported.
	Expression string `json:"expression,omitempty"`

	// Location: An optional string indicating the location of the
	// expression for error
	// reporting, e.g. a file name and a position in the file.
	Location string `json:"location,omitempty"`

	// Title: An optional title for the expression, i.e. a short string
	// describing
	// its purpose. This can be used e.g. in UIs which allow to enter
	// the
	// expression.
	Title string `json:"title,omitempty"`

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

func (s *Expr) MarshalJSON() ([]byte, error) {
	type noMethod Expr
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Field: A single field of a message type.
type Field struct {
	// Cardinality: The field cardinality.
	//
	// Possible values:
	//   "CARDINALITY_UNKNOWN" - For fields with unknown cardinality.
	//   "CARDINALITY_OPTIONAL" - For optional fields.
	//   "CARDINALITY_REQUIRED" - For required fields. Proto2 syntax only.
	//   "CARDINALITY_REPEATED" - For repeated fields.
	Cardinality string `json:"cardinality,omitempty"`

	// DefaultValue: The string value of the default value of this field.
	// Proto2 syntax only.
	DefaultValue string `json:"defaultValue,omitempty"`

	// JsonName: The field JSON name.
	JsonName string `json:"jsonName,omitempty"`

	// Kind: The field type.
	//
	// Possible values:
	//   "TYPE_UNKNOWN" - Field type unknown.
	//   "TYPE_DOUBLE" - Field type double.
	//   "TYPE_FLOAT" - Field type float.
	//   "TYPE_INT64" - Field type int64.
	//   "TYPE_UINT64" - Field type uint64.
	//   "TYPE_INT32" - Field type int32.
	//   "TYPE_FIXED64" - Field type fixed64.
	//   "TYPE_FIXED32" - Field type fixed32.
	//   "TYPE_BOOL" - Field type bool.
	//   "TYPE_STRING" - Field type string.
	//   "TYPE_GROUP" - Field type group. Proto2 syntax only, and
	// deprecated.
	//   "TYPE_MESSAGE" - Field type message.
	//   "TYPE_BYTES" - Field type bytes.
	//   "TYPE_UINT32" - Field type uint32.
	//   "TYPE_ENUM" - Field type enum.
	//   "TYPE_SFIXED32" - Field type sfixed32.
	//   "TYPE_SFIXED64" - Field type sfixed64.
	//   "TYPE_SINT32" - Field type sint32.
	//   "TYPE_SINT64" - Field type sint64.
	Kind string `json:"kind,omitempty"`

	// Name: The field name.
	Name string `json:"name,omitempty"`

	// Number: The field number.
	Number int64 `json:"number,omitempty"`

	// OneofIndex: The index of the field type in `Type.oneofs`, for message
	// or enumeration
	// types. The first type has index 1; zero means the type is not in the
	// list.
	OneofIndex int64 `json:"oneofIndex,omitempty"`

	// Options: The protocol buffer options.
	Options []*Option `json:"options,omitempty"`

	// Packed: Whether to use alternative packed wire representation.
	Packed bool `json:"packed,omitempty"`

	// TypeUrl: The field type URL, without the scheme, for message or
	// enumeration
	// types. Example: "type.googleapis.com/google.protobuf.Timestamp".
	TypeUrl string `json:"typeUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cardinality") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cardinality") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Field) MarshalJSON() ([]byte, error) {
	type noMethod Field
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FlowOperationMetadata: The metadata associated with a long running
// operation resource.
type FlowOperationMetadata struct {
	// CancelState: The state of the operation with respect to cancellation.
	//
	// Possible values:
	//   "RUNNING" - Default state, cancellable but not cancelled.
	//   "UNCANCELLABLE" - The operation has proceeded past the point of no
	// return and cannot
	// be cancelled.
	//   "CANCELLED" - The operation has been cancelled, work should
	// cease
	// and any needed rollback steps executed.
	CancelState string `json:"cancelState,omitempty"`

	// Deadline: Deadline for the flow to complete, to prevent orphaned
	// Operations.
	//
	// If the flow has not completed by this time, it may be terminated
	// by
	// the engine, or force-failed by Operation lookup.
	//
	// Note that this is not a hard deadline after which the Flow
	// will
	// definitely be failed, rather it is a deadline after which it is
	// reasonable
	// to suspect a problem and other parts of the system may kill
	// operation
	// to ensure we don't have orphans.
	// see also: go/prevent-orphaned-operations
	Deadline string `json:"deadline,omitempty"`

	// FlowName: The name of the top-level flow corresponding to this
	// operation.
	// Must be equal to the "name" field for a FlowName enum.
	FlowName string `json:"flowName,omitempty"`

	// ResourceNames: The full name of the resources that this flow is
	// directly associated with.
	ResourceNames []string `json:"resourceNames,omitempty"`

	// StartTime: The start time of the operation.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CancelState") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CancelState") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FlowOperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod FlowOperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GenerateConfigReportRequest: Request message for GenerateConfigReport
// method.
type GenerateConfigReportRequest struct {
	// NewConfig: Service configuration for which we want to generate the
	// report.
	// For this version of API, the supported types
	// are
	// google.api.servicemanagement.v1.ConfigRef,
	// google.api.servicemanag
	// ement.v1.ConfigSource,
	// and google.api.Service
	NewConfig googleapi.RawMessage `json:"newConfig,omitempty"`

	// OldConfig: Service configuration against which the comparison will be
	// done.
	// For this version of API, the supported types
	// are
	// google.api.servicemanagement.v1.ConfigRef,
	// google.api.servicemanag
	// ement.v1.ConfigSource,
	// and google.api.Service
	OldConfig googleapi.RawMessage `json:"oldConfig,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NewConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NewConfig") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GenerateConfigReportRequest) MarshalJSON() ([]byte, error) {
	type noMethod GenerateConfigReportRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GenerateConfigReportResponse: Response message for
// GenerateConfigReport method.
type GenerateConfigReportResponse struct {
	// ChangeReports: list of ChangeReport, each corresponding to comparison
	// between two
	// service configurations.
	ChangeReports []*ChangeReport `json:"changeReports,omitempty"`

	// Diagnostics: Errors / Linter warnings associated with the service
	// definition this
	// report
	// belongs to.
	Diagnostics []*Diagnostic `json:"diagnostics,omitempty"`

	// Id: ID of the service configuration this report belongs to.
	Id string `json:"id,omitempty"`

	// ServiceName: Name of the service this report belongs to.
	ServiceName string `json:"serviceName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ChangeReports") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChangeReports") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GenerateConfigReportResponse) MarshalJSON() ([]byte, error) {
	type noMethod GenerateConfigReportResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetIamPolicyRequest: Request message for `GetIamPolicy` method.
type GetIamPolicyRequest struct {
}

// Http: Defines the HTTP configuration for an API service. It contains
// a list of
// HttpRule, each specifying the mapping of an RPC method
// to one or more HTTP REST API methods.
type Http struct {
	// FullyDecodeReservedExpansion: When set to true, URL path parmeters
	// will be fully URI-decoded except in
	// cases of single segment matches in reserved expansion, where "%2F"
	// will be
	// left encoded.
	//
	// The default behavior is to not decode RFC 6570 reserved characters in
	// multi
	// segment matches.
	FullyDecodeReservedExpansion bool `json:"fullyDecodeReservedExpansion,omitempty"`

	// Rules: A list of HTTP configuration rules that apply to individual
	// API methods.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*HttpRule `json:"rules,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "FullyDecodeReservedExpansion") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "FullyDecodeReservedExpansion") to include in API requests with the
	// JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Http) MarshalJSON() ([]byte, error) {
	type noMethod Http
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HttpRule: `HttpRule` defines the mapping of an RPC method to one or
// more HTTP
// REST API methods. The mapping specifies how different portions of the
// RPC
// request message are mapped to URL path, URL query parameters,
// and
// HTTP request body. The mapping is typically specified as
// an
// `google.api.http` annotation on the RPC method,
// see "google/api/annotations.proto" for details.
//
// The mapping consists of a field specifying the path template
// and
// method kind.  The path template can refer to fields in the
// request
// message, as in the example below which describes a REST GET
// operation on a resource collection of messages:
//
//
//     service Messaging {
//       rpc GetMessage(GetMessageRequest) returns (Message) {
//         option (google.api.http).get =
// "/v1/messages/{message_id}/{sub.subfield}";
//       }
//     }
//     message GetMessageRequest {
//       message SubMessage {
//         string subfield = 1;
//       }
//       string message_id = 1; // mapped to the URL
//       SubMessage sub = 2;    // `sub.subfield` is url-mapped
//     }
//     message Message {
//       string text = 1; // content of the resource
//     }
//
// The same http annotation can alternatively be expressed inside
// the
// `GRPC API Configuration` YAML file.
//
//     http:
//       rules:
//         - selector: <proto_package_name>.Messaging.GetMessage
//           get: /v1/messages/{message_id}/{sub.subfield}
//
// This definition enables an automatic, bidrectional mapping of
// HTTP
// JSON to RPC. Example:
//
// HTTP | RPC
// -----|-----
// `GET /v1/messages/123456/foo`  | `GetMessage(message_id: "123456"
// sub: SubMessage(subfield: "foo"))`
//
// In general, not only fields but also field paths can be
// referenced
// from a path pattern. Fields mapped to the path pattern cannot
// be
// repeated and must have a primitive (non-message) type.
//
// Any fields in the request message which are not bound by the
// path
// pattern automatically become (optional) HTTP query
// parameters. Assume the following definition of the request
// message:
//
//
//     service Messaging {
//       rpc GetMessage(GetMessageRequest) returns (Message) {
//         option (google.api.http).get = "/v1/messages/{message_id}";
//       }
//     }
//     message GetMessageRequest {
//       message SubMessage {
//         string subfield = 1;
//       }
//       string message_id = 1; // mapped to the URL
//       int64 revision = 2;    // becomes a parameter
//       SubMessage sub = 3;    // `sub.subfield` becomes a parameter
//     }
//
//
// This enables a HTTP JSON to RPC mapping as below:
//
// HTTP | RPC
// -----|-----
// `GET /v1/messages/123456?revision=2&sub.subfield=foo` |
// `GetMessage(message_id: "123456" revision: 2 sub:
// SubMessage(subfield: "foo"))`
//
// Note that fields which are mapped to HTTP parameters must have
// a
// primitive type or a repeated primitive type. Message types are
// not
// allowed. In the case of a repeated type, the parameter can
// be
// repeated in the URL, as in `...?param=A&param=B`.
//
// For HTTP method kinds which allow a request body, the `body`
// field
// specifies the mapping. Consider a REST update method on the
// message resource collection:
//
//
//     service Messaging {
//       rpc UpdateMessage(UpdateMessageRequest) returns (Message) {
//         option (google.api.http) = {
//           put: "/v1/messages/{message_id}"
//           body: "message"
//         };
//       }
//     }
//     message UpdateMessageRequest {
//       string message_id = 1; // mapped to the URL
//       Message message = 2;   // mapped to the body
//     }
//
//
// The following HTTP JSON to RPC mapping is enabled, where
// the
// representation of the JSON in the request body is determined
// by
// protos JSON encoding:
//
// HTTP | RPC
// -----|-----
// `PUT /v1/messages/123456 { "text": "Hi!" }` |
// `UpdateMessage(message_id: "123456" message { text: "Hi!" })`
//
// The special name `*` can be used in the body mapping to define
// that
// every field not bound by the path template should be mapped to
// the
// request body.  This enables the following alternative definition
// of
// the update method:
//
//     service Messaging {
//       rpc UpdateMessage(Message) returns (Message) {
//         option (google.api.http) = {
//           put: "/v1/messages/{message_id}"
//           body: "*"
//         };
//       }
//     }
//     message Message {
//       string message_id = 1;
//       string text = 2;
//     }
//
//
// The following HTTP JSON to RPC mapping is enabled:
//
// HTTP | RPC
// -----|-----
// `PUT /v1/messages/123456 { "text": "Hi!" }` |
// `UpdateMessage(message_id: "123456" text: "Hi!")`
//
// Note that when using `*` in the body mapping, it is not possible
// to
// have HTTP parameters, as all fields not bound by the path end in
// the body. This makes this option more rarely used in practice
// of
// defining REST APIs. The common usage of `*` is in custom
// methods
// which don't use the URL at all for transferring data.
//
// It is possible to define multiple HTTP methods for one RPC by
// using
// the `additional_bindings` option. Example:
//
//     service Messaging {
//       rpc GetMessage(GetMessageRequest) returns (Message) {
//         option (google.api.http) = {
//           get: "/v1/messages/{message_id}"
//           additional_bindings {
//             get: "/v1/users/{user_id}/messages/{message_id}"
//           }
//         };
//       }
//     }
//     message GetMessageRequest {
//       string message_id = 1;
//       string user_id = 2;
//     }
//
//
// This enables the following two alternative HTTP JSON to
// RPC
// mappings:
//
// HTTP | RPC
// -----|-----
// `GET /v1/messages/123456` | `GetMessage(message_id: "123456")`
// `GET /v1/users/me/messages/123456` | `GetMessage(user_id: "me"
// message_id: "123456")`
//
// # Rules for HTTP mapping
//
// The rules for mapping HTTP path, query parameters, and body fields
// to the request message are as follows:
//
// 1. The `body` field specifies either `*` or a field path, or is
//    omitted. If omitted, it indicates there is no HTTP request
// body.
// 2. Leaf fields (recursive expansion of nested messages in the
//    request) can be classified into three types:
//     (a) Matched in the URL template.
//     (b) Covered by body (if body is `*`, everything except (a)
// fields;
//         else everything under the body field)
//     (c) All other fields.
// 3. URL query parameters found in the HTTP request are mapped to (c)
// fields.
// 4. Any body sent with an HTTP request can contain only (b)
// fields.
//
// The syntax of the path template is as follows:
//
//     Template = "/" Segments [ Verb ] ;
//     Segments = Segment { "/" Segment } ;
//     Segment  = "*" | "**" | LITERAL | Variable ;
//     Variable = "{" FieldPath [ "=" Segments ] "}" ;
//     FieldPath = IDENT { "." IDENT } ;
//     Verb     = ":" LITERAL ;
//
// The syntax `*` matches a single path segment. The syntax `**` matches
// zero
// or more path segments, which must be the last part of the path except
// the
// `Verb`. The syntax `LITERAL` matches literal text in the path.
//
// The syntax `Variable` matches part of the URL path as specified by
// its
// template. A variable template must not contain other variables. If a
// variable
// matches a single path segment, its template may be omitted, e.g.
// `{var}`
// is equivalent to `{var=*}`.
//
// If a variable contains exactly one path segment, such as "{var}"
// or
// "{var=*}", when such a variable is expanded into a URL path, all
// characters
// except `[-_.~0-9a-zA-Z]` are percent-encoded. Such variables show up
// in the
// Discovery Document as `{var}`.
//
// If a variable contains one or more path segments, such as
// "{var=foo/*}"
// or "{var=**}", when such a variable is expanded into a URL path,
// all
// characters except `[-_.~/0-9a-zA-Z]` are percent-encoded. Such
// variables
// show up in the Discovery Document as `{+var}`.
//
// NOTE: While the single segment variable matches the semantics of
// [RFC 6570](https://tools.ietf.org/html/rfc6570) Section 3.2.2
// Simple String Expansion, the multi segment variable **does not**
// match
// RFC 6570 Reserved Expansion. The reason is that the Reserved
// Expansion
// does not expand special characters like `?` and `#`, which would
// lead
// to invalid URLs.
//
// NOTE: the field paths in variables and in the `body` must not refer
// to
// repeated fields or map fields.
type HttpRule struct {
	// AdditionalBindings: Additional HTTP bindings for the selector. Nested
	// bindings must
	// not contain an `additional_bindings` field themselves (that is,
	// the nesting may only be one level deep).
	AdditionalBindings []*HttpRule `json:"additionalBindings,omitempty"`

	// Body: The name of the request field whose value is mapped to the HTTP
	// body, or
	// `*` for mapping all fields not captured by the path pattern to the
	// HTTP
	// body. NOTE: the referred field must not be a repeated field and must
	// be
	// present at the top-level of request message type.
	Body string `json:"body,omitempty"`

	// Custom: The custom pattern is used for specifying an HTTP method that
	// is not
	// included in the `pattern` field, such as HEAD, or "*" to leave
	// the
	// HTTP method unspecified for this rule. The wild-card rule is
	// useful
	// for services that provide content to Web (HTML) clients.
	Custom *CustomHttpPattern `json:"custom,omitempty"`

	// Delete: Used for deleting a resource.
	Delete string `json:"delete,omitempty"`

	// Get: Used for listing and getting information about resources.
	Get string `json:"get,omitempty"`

	// MediaDownload: Use this only for Scotty Requests. Do not use this for
	// bytestream methods.
	// For media support, add instead [][google.bytestream.RestByteStream]
	// as an
	// API to your configuration.
	MediaDownload *MediaDownload `json:"mediaDownload,omitempty"`

	// MediaUpload: Use this only for Scotty Requests. Do not use this for
	// media support using
	// Bytestream, add instead
	// [][google.bytestream.RestByteStream] as an API to your
	// configuration for Bytestream methods.
	MediaUpload *MediaUpload `json:"mediaUpload,omitempty"`

	// Patch: Used for updating a resource.
	Patch string `json:"patch,omitempty"`

	// Post: Used for creating a resource.
	Post string `json:"post,omitempty"`

	// Put: Used for updating a resource.
	Put string `json:"put,omitempty"`

	// ResponseBody: The name of the response field whose value is mapped to
	// the HTTP body of
	// response. Other response fields are ignored. This field is optional.
	// When
	// not set, the response message will be used as HTTP body of
	// response.
	// NOTE: the referred field must be not a repeated field and must be
	// present
	// at the top-level of response message type.
	ResponseBody string `json:"responseBody,omitempty"`

	// Selector: Selects methods to which this rule applies.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdditionalBindings")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdditionalBindings") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *HttpRule) MarshalJSON() ([]byte, error) {
	type noMethod HttpRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LabelDescriptor: A description of a label.
type LabelDescriptor struct {
	// Description: A human-readable description for the label.
	Description string `json:"description,omitempty"`

	// Key: The label key.
	Key string `json:"key,omitempty"`

	// ValueType: The type of data that can be assigned to the label.
	//
	// Possible values:
	//   "STRING" - A variable-length string. This is the default.
	//   "BOOL" - Boolean; true or false.
	//   "INT64" - A 64-bit signed integer.
	ValueType string `json:"valueType,omitempty"`

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

func (s *LabelDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod LabelDescriptor
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

// ListServiceConfigsResponse: Response message for ListServiceConfigs
// method.
type ListServiceConfigsResponse struct {
	// NextPageToken: The token of the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServiceConfigs: The list of service configuration resources.
	ServiceConfigs []*Service `json:"serviceConfigs,omitempty"`

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

func (s *ListServiceConfigsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListServiceConfigsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListServiceRolloutsResponse: Response message for ListServiceRollouts
// method.
type ListServiceRolloutsResponse struct {
	// NextPageToken: The token of the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Rollouts: The list of rollout resources.
	Rollouts []*Rollout `json:"rollouts,omitempty"`

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

func (s *ListServiceRolloutsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListServiceRolloutsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListServicesResponse: Response message for `ListServices` method.
type ListServicesResponse struct {
	// NextPageToken: Token that can be passed to `ListServices` to resume a
	// paginated query.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Services: The returned services will only have the name field set.
	Services []*ManagedService `json:"services,omitempty"`

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

func (s *ListServicesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListServicesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogDescriptor: A description of a log type. Example in YAML format:
//
//     - name: library.googleapis.com/activity_history
//       description: The history of borrowing and returning library
// items.
//       display_name: Activity
//       labels:
//       - key: /customer_id
//         description: Identifier of a library customer
type LogDescriptor struct {
	// Description: A human-readable description of this log. This
	// information appears in
	// the documentation and can contain details.
	Description string `json:"description,omitempty"`

	// DisplayName: The human-readable name for this log. This information
	// appears on
	// the user interface and should be concise.
	DisplayName string `json:"displayName,omitempty"`

	// Labels: The set of labels that are available to describe a specific
	// log entry.
	// Runtime requests that contain labels not specified here
	// are
	// considered invalid.
	Labels []*LabelDescriptor `json:"labels,omitempty"`

	// Name: The name of the log. It must be less than 512 characters long
	// and can
	// include the following characters: upper- and lower-case
	// alphanumeric
	// characters [A-Za-z0-9], and punctuation characters including
	// slash, underscore, hyphen, period [/_-.].
	Name string `json:"name,omitempty"`

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

func (s *LogDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod LogDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Logging: Logging configuration of the service.
//
// The following example shows how to configure logs to be sent to
// the
// producer and consumer projects. In the example, the
// `activity_history`
// log is sent to both the producer and consumer projects, whereas
// the
// `purchase_history` log is only sent to the producer project.
//
//     monitored_resources:
//     - type: library.googleapis.com/branch
//       labels:
//       - key: /city
//         description: The city where the library branch is located
// in.
//       - key: /name
//         description: The name of the branch.
//     logs:
//     - name: activity_history
//       labels:
//       - key: /customer_id
//     - name: purchase_history
//     logging:
//       producer_destinations:
//       - monitored_resource: library.googleapis.com/branch
//         logs:
//         - activity_history
//         - purchase_history
//       consumer_destinations:
//       - monitored_resource: library.googleapis.com/branch
//         logs:
//         - activity_history
type Logging struct {
	// ConsumerDestinations: Logging configurations for sending logs to the
	// consumer project.
	// There can be multiple consumer destinations, each one must have
	// a
	// different monitored resource type. A log can be used in at most
	// one consumer destination.
	ConsumerDestinations []*LoggingDestination `json:"consumerDestinations,omitempty"`

	// ProducerDestinations: Logging configurations for sending logs to the
	// producer project.
	// There can be multiple producer destinations, each one must have
	// a
	// different monitored resource type. A log can be used in at most
	// one producer destination.
	ProducerDestinations []*LoggingDestination `json:"producerDestinations,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ConsumerDestinations") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConsumerDestinations") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Logging) MarshalJSON() ([]byte, error) {
	type noMethod Logging
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LoggingDestination: Configuration of a specific logging destination
// (the producer project
// or the consumer project).
type LoggingDestination struct {
	// Logs: Names of the logs to be sent to this destination. Each name
	// must
	// be defined in the Service.logs section. If the log name is
	// not a domain scoped name, it will be automatically prefixed with
	// the service name followed by "/".
	Logs []string `json:"logs,omitempty"`

	// MonitoredResource: The monitored resource type. The type must be
	// defined in the
	// Service.monitored_resources section.
	MonitoredResource string `json:"monitoredResource,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Logs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Logs") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LoggingDestination) MarshalJSON() ([]byte, error) {
	type noMethod LoggingDestination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedService: The full representation of a Service that is managed
// by
// Google Service Management.
type ManagedService struct {
	// ProducerProjectId: ID of the project that produces and owns this
	// service.
	ProducerProjectId string `json:"producerProjectId,omitempty"`

	// ServiceName: The name of the service. See the
	// [overview](/service-management/overview)
	// for naming requirements.
	ServiceName string `json:"serviceName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ProducerProjectId")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProducerProjectId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ManagedService) MarshalJSON() ([]byte, error) {
	type noMethod ManagedService
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MediaDownload: Defines the Media configuration for a service in case
// of a download.
// Use this only for Scotty Requests. Do not use this for media support
// using
// Bytestream, add instead [][google.bytestream.RestByteStream] as an
// API to
// your configuration for Bytestream methods.
type MediaDownload struct {
	// CompleteNotification: A boolean that determines whether a
	// notification for the completion of a
	// download should be sent to the backend.
	CompleteNotification bool `json:"completeNotification,omitempty"`

	// DownloadService: DO NOT USE FIELDS BELOW THIS LINE UNTIL THIS WARNING
	// IS REMOVED.
	//
	// Specify name of the download service if one is used for download.
	DownloadService string `json:"downloadService,omitempty"`

	// Dropzone: Name of the Scotty dropzone to use for the current API.
	Dropzone string `json:"dropzone,omitempty"`

	// Enabled: Whether download is enabled.
	Enabled bool `json:"enabled,omitempty"`

	// MaxDirectDownloadSize: Optional maximum acceptable size for direct
	// download.
	// The size is specified in bytes.
	MaxDirectDownloadSize int64 `json:"maxDirectDownloadSize,omitempty,string"`

	// UseDirectDownload: A boolean that determines if direct download from
	// ESF should be used for
	// download of this media.
	UseDirectDownload bool `json:"useDirectDownload,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "CompleteNotification") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompleteNotification") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MediaDownload) MarshalJSON() ([]byte, error) {
	type noMethod MediaDownload
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MediaUpload: Defines the Media configuration for a service in case of
// an upload.
// Use this only for Scotty Requests. Do not use this for media support
// using
// Bytestream, add instead [][google.bytestream.RestByteStream] as an
// API to
// your configuration for Bytestream methods.
type MediaUpload struct {
	// CompleteNotification: A boolean that determines whether a
	// notification for the completion of an
	// upload should be sent to the backend. These notifications will not be
	// seen
	// by the client and will not consume quota.
	CompleteNotification bool `json:"completeNotification,omitempty"`

	// Dropzone: Name of the Scotty dropzone to use for the current API.
	Dropzone string `json:"dropzone,omitempty"`

	// Enabled: Whether upload is enabled.
	Enabled bool `json:"enabled,omitempty"`

	// MaxSize: Optional maximum acceptable size for an upload.
	// The size is specified in bytes.
	MaxSize int64 `json:"maxSize,omitempty,string"`

	// MimeTypes: An array of mimetype patterns. Esf will only accept
	// uploads that match one
	// of the given patterns.
	MimeTypes []string `json:"mimeTypes,omitempty"`

	// ProgressNotification: Whether to receive a notification for progress
	// changes of media upload.
	ProgressNotification bool `json:"progressNotification,omitempty"`

	// StartNotification: Whether to receive a notification on the start of
	// media upload.
	StartNotification bool `json:"startNotification,omitempty"`

	// UploadService: DO NOT USE FIELDS BELOW THIS LINE UNTIL THIS WARNING
	// IS REMOVED.
	//
	// Specify name of the upload service if one is used for upload.
	UploadService string `json:"uploadService,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "CompleteNotification") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompleteNotification") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MediaUpload) MarshalJSON() ([]byte, error) {
	type noMethod MediaUpload
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Method: Method represents a method of an API interface.
type Method struct {
	// Name: The simple name of this method.
	Name string `json:"name,omitempty"`

	// Options: Any metadata attached to the method.
	Options []*Option `json:"options,omitempty"`

	// RequestStreaming: If true, the request is streamed.
	RequestStreaming bool `json:"requestStreaming,omitempty"`

	// RequestTypeUrl: A URL of the input message type.
	RequestTypeUrl string `json:"requestTypeUrl,omitempty"`

	// ResponseStreaming: If true, the response is streamed.
	ResponseStreaming bool `json:"responseStreaming,omitempty"`

	// ResponseTypeUrl: The URL of the output message type.
	ResponseTypeUrl string `json:"responseTypeUrl,omitempty"`

	// Syntax: The source syntax of this method.
	//
	// Possible values:
	//   "SYNTAX_PROTO2" - Syntax `proto2`.
	//   "SYNTAX_PROTO3" - Syntax `proto3`.
	Syntax string `json:"syntax,omitempty"`

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

func (s *Method) MarshalJSON() ([]byte, error) {
	type noMethod Method
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricDescriptor: Defines a metric type and its schema. Once a metric
// descriptor is created,
// deleting or altering it stops data collection and makes the metric
// type's
// existing data unusable.
type MetricDescriptor struct {
	// Description: A detailed description of the metric, which can be used
	// in documentation.
	Description string `json:"description,omitempty"`

	// DisplayName: A concise name for the metric, which can be displayed in
	// user interfaces.
	// Use sentence case without an ending period, for example "Request
	// count".
	DisplayName string `json:"displayName,omitempty"`

	// Labels: The set of labels that can be used to describe a
	// specific
	// instance of this metric type. For example,
	// the
	// `appengine.googleapis.com/http/server/response_latencies` metric
	// type has a label for the HTTP response code, `response_code`, so
	// you can look at latencies for successful responses or just
	// for responses that failed.
	Labels []*LabelDescriptor `json:"labels,omitempty"`

	// MetricKind: Whether the metric records instantaneous values, changes
	// to a value, etc.
	// Some combinations of `metric_kind` and `value_type` might not be
	// supported.
	//
	// Possible values:
	//   "METRIC_KIND_UNSPECIFIED" - Do not use this default value.
	//   "GAUGE" - An instantaneous measurement of a value.
	//   "DELTA" - The change in a value during a time interval.
	//   "CUMULATIVE" - A value accumulated over a time interval.
	// Cumulative
	// measurements in a time series should have the same start time
	// and increasing end times, until an event resets the cumulative
	// value to zero and sets a new start time for the following
	// points.
	MetricKind string `json:"metricKind,omitempty"`

	// Name: The resource name of the metric descriptor. Depending on
	// the
	// implementation, the name typically includes: (1) the parent resource
	// name
	// that defines the scope of the metric type or of its data; and (2)
	// the
	// metric's URL-encoded type, which also appears in the `type` field of
	// this
	// descriptor. For example, following is the resource name of a
	// custom
	// metric within the GCP project `my-project-id`:
	//
	//
	// "projects/my-project-id/metricDescriptors/custom.googleapis.com%2Finvo
	// ice%2Fpaid%2Famount"
	Name string `json:"name,omitempty"`

	// Type: The metric type, including its DNS name prefix. The type is
	// not
	// URL-encoded.  All user-defined custom metric types have the DNS
	// name
	// `custom.googleapis.com`.  Metric types should use a natural
	// hierarchical
	// grouping. For example:
	//
	//     "custom.googleapis.com/invoice/paid/amount"
	//     "appengine.googleapis.com/http/server/response_latencies"
	Type string `json:"type,omitempty"`

	// Unit: The unit in which the metric value is reported. It is only
	// applicable
	// if the `value_type` is `INT64`, `DOUBLE`, or `DISTRIBUTION`.
	// The
	// supported units are a subset of [The Unified Code for Units
	// of
	// Measure](http://unitsofmeasure.org/ucum.html) standard:
	//
	// **Basic units (UNIT)**
	//
	// * `bit`   bit
	// * `By`    byte
	// * `s`     second
	// * `min`   minute
	// * `h`     hour
	// * `d`     day
	//
	// **Prefixes (PREFIX)**
	//
	// * `k`     kilo    (10**3)
	// * `M`     mega    (10**6)
	// * `G`     giga    (10**9)
	// * `T`     tera    (10**12)
	// * `P`     peta    (10**15)
	// * `E`     exa     (10**18)
	// * `Z`     zetta   (10**21)
	// * `Y`     yotta   (10**24)
	// * `m`     milli   (10**-3)
	// * `u`     micro   (10**-6)
	// * `n`     nano    (10**-9)
	// * `p`     pico    (10**-12)
	// * `f`     femto   (10**-15)
	// * `a`     atto    (10**-18)
	// * `z`     zepto   (10**-21)
	// * `y`     yocto   (10**-24)
	// * `Ki`    kibi    (2**10)
	// * `Mi`    mebi    (2**20)
	// * `Gi`    gibi    (2**30)
	// * `Ti`    tebi    (2**40)
	//
	// **Grammar**
	//
	// The grammar includes the dimensionless unit `1`, such as `1/s`.
	//
	// The grammar also includes these connectors:
	//
	// * `/`    division (as an infix operator, e.g. `1/s`).
	// * `.`    multiplication (as an infix operator, e.g. `GBy.d`)
	//
	// The grammar for a unit is as follows:
	//
	//     Expression = Component { "." Component } { "/" Component } ;
	//
	//     Component = [ PREFIX ] UNIT [ Annotation ]
	//               | Annotation
	//               | "1"
	//               ;
	//
	//     Annotation = "{" NAME "}" ;
	//
	// Notes:
	//
	// * `Annotation` is just a comment if it follows a `UNIT` and is
	//    equivalent to `1` if it is used alone. For examples,
	//    `{requests}/s == 1/s`, `By{transmitted}/s == By/s`.
	// * `NAME` is a sequence of non-blank printable ASCII characters not
	//    containing '{' or '}'.
	Unit string `json:"unit,omitempty"`

	// ValueType: Whether the measurement is an integer, a floating-point
	// number, etc.
	// Some combinations of `metric_kind` and `value_type` might not be
	// supported.
	//
	// Possible values:
	//   "VALUE_TYPE_UNSPECIFIED" - Do not use this default value.
	//   "BOOL" - The value is a boolean.
	// This value type can be used only if the metric kind is `GAUGE`.
	//   "INT64" - The value is a signed 64-bit integer.
	//   "DOUBLE" - The value is a double precision floating point number.
	//   "STRING" - The value is a text string.
	// This value type can be used only if the metric kind is `GAUGE`.
	//   "DISTRIBUTION" - The value is a `Distribution`.
	//   "MONEY" - The value is money.
	ValueType string `json:"valueType,omitempty"`

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

// MetricRule: Bind API methods to metrics. Binding a method to a metric
// causes that
// metric's configured quota behaviors to apply to the method call.
type MetricRule struct {
	// MetricCosts: Metrics to update when the selected methods are called,
	// and the associated
	// cost applied to each metric.
	//
	// The key of the map is the metric name, and the values are the
	// amount
	// increased for the metric against which the quota limits are
	// defined.
	// The value must not be negative.
	MetricCosts map[string]string `json:"metricCosts,omitempty"`

	// Selector: Selects the methods to which this rule applies.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MetricCosts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MetricCosts") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricRule) MarshalJSON() ([]byte, error) {
	type noMethod MetricRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Mixin: Declares an API Interface to be included in this interface.
// The including
// interface must redeclare all the methods from the included interface,
// but
// documentation and options are inherited as follows:
//
// - If after comment and whitespace stripping, the documentation
//   string of the redeclared method is empty, it will be inherited
//   from the original method.
//
// - Each annotation belonging to the service config (http,
//   visibility) which is not set in the redeclared method will be
//   inherited.
//
// - If an http annotation is inherited, the path pattern will be
//   modified as follows. Any version prefix will be replaced by the
//   version of the including interface plus the root path if
//   specified.
//
// Example of a simple mixin:
//
//     package google.acl.v1;
//     service AccessControl {
//       // Get the underlying ACL object.
//       rpc GetAcl(GetAclRequest) returns (Acl) {
//         option (google.api.http).get = "/v1/{resource=**}:getAcl";
//       }
//     }
//
//     package google.storage.v2;
//     service Storage {
//       //       rpc GetAcl(GetAclRequest) returns (Acl);
//
//       // Get a data record.
//       rpc GetData(GetDataRequest) returns (Data) {
//         option (google.api.http).get = "/v2/{resource=**}";
//       }
//     }
//
// Example of a mixin configuration:
//
//     apis:
//     - name: google.storage.v2.Storage
//       mixins:
//       - name: google.acl.v1.AccessControl
//
// The mixin construct implies that all methods in `AccessControl`
// are
// also declared with same name and request/response types in
// `Storage`. A documentation generator or annotation processor will
// see the effective `Storage.GetAcl` method after
// inherting
// documentation and annotations as follows:
//
//     service Storage {
//       // Get the underlying ACL object.
//       rpc GetAcl(GetAclRequest) returns (Acl) {
//         option (google.api.http).get = "/v2/{resource=**}:getAcl";
//       }
//       ...
//     }
//
// Note how the version in the path pattern changed from `v1` to
// `v2`.
//
// If the `root` field in the mixin is specified, it should be
// a
// relative path under which inherited HTTP paths are placed. Example:
//
//     apis:
//     - name: google.storage.v2.Storage
//       mixins:
//       - name: google.acl.v1.AccessControl
//         root: acls
//
// This implies the following inherited HTTP annotation:
//
//     service Storage {
//       // Get the underlying ACL object.
//       rpc GetAcl(GetAclRequest) returns (Acl) {
//         option (google.api.http).get =
// "/v2/acls/{resource=**}:getAcl";
//       }
//       ...
//     }
type Mixin struct {
	// Name: The fully qualified name of the interface which is included.
	Name string `json:"name,omitempty"`

	// Root: If non-empty specifies a path under which inherited HTTP
	// paths
	// are rooted.
	Root string `json:"root,omitempty"`

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

func (s *Mixin) MarshalJSON() ([]byte, error) {
	type noMethod Mixin
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MonitoredResourceDescriptor: An object that describes the schema of a
// MonitoredResource object using a
// type name and a set of labels.  For example, the monitored
// resource
// descriptor for Google Compute Engine VM instances has a type
// of
// "gce_instance" and specifies the use of the labels "instance_id"
// and
// "zone" to identify particular VM instances.
//
// Different APIs can support different monitored resource types. APIs
// generally
// provide a `list` method that returns the monitored resource
// descriptors used
// by the API.
type MonitoredResourceDescriptor struct {
	// Description: Optional. A detailed description of the monitored
	// resource type that might
	// be used in documentation.
	Description string `json:"description,omitempty"`

	// DisplayName: Optional. A concise name for the monitored resource type
	// that might be
	// displayed in user interfaces. It should be a Title Cased Noun
	// Phrase,
	// without any article or other determiners. For example,
	// "Google Cloud SQL Database".
	DisplayName string `json:"displayName,omitempty"`

	// Labels: Required. A set of labels used to describe instances of this
	// monitored
	// resource type. For example, an individual Google Cloud SQL database
	// is
	// identified by values for the labels "database_id" and "zone".
	Labels []*LabelDescriptor `json:"labels,omitempty"`

	// Name: Optional. The resource name of the monitored resource
	// descriptor:
	// "projects/{project_id}/monitoredResourceDescriptors/{type
	// }" where
	// {type} is the value of the `type` field in this object
	// and
	// {project_id} is a project ID that provides API-specific context
	// for
	// accessing the type.  APIs that do not use project information can use
	// the
	// resource name format "monitoredResourceDescriptors/{type}".
	Name string `json:"name,omitempty"`

	// Type: Required. The monitored resource type. For example, the
	// type
	// "cloudsql_database" represents databases in Google Cloud SQL.
	// The maximum length of this value is 256 characters.
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

func (s *MonitoredResourceDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod MonitoredResourceDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Monitoring: Monitoring configuration of the service.
//
// The example below shows how to configure monitored resources and
// metrics
// for monitoring. In the example, a monitored resource and two metrics
// are
// defined. The `library.googleapis.com/book/returned_count` metric is
// sent
// to both producer and consumer projects, whereas
// the
// `library.googleapis.com/book/overdue_count` metric is only sent to
// the
// consumer project.
//
//     monitored_resources:
//     - type: library.googleapis.com/branch
//       labels:
//       - key: /city
//         description: The city where the library branch is located
// in.
//       - key: /name
//         description: The name of the branch.
//     metrics:
//     - name: library.googleapis.com/book/returned_count
//       metric_kind: DELTA
//       value_type: INT64
//       labels:
//       - key: /customer_id
//     - name: library.googleapis.com/book/overdue_count
//       metric_kind: GAUGE
//       value_type: INT64
//       labels:
//       - key: /customer_id
//     monitoring:
//       producer_destinations:
//       - monitored_resource: library.googleapis.com/branch
//         metrics:
//         - library.googleapis.com/book/returned_count
//       consumer_destinations:
//       - monitored_resource: library.googleapis.com/branch
//         metrics:
//         - library.googleapis.com/book/returned_count
//         - library.googleapis.com/book/overdue_count
type Monitoring struct {
	// ConsumerDestinations: Monitoring configurations for sending metrics
	// to the consumer project.
	// There can be multiple consumer destinations, each one must have
	// a
	// different monitored resource type. A metric can be used in at
	// most
	// one consumer destination.
	ConsumerDestinations []*MonitoringDestination `json:"consumerDestinations,omitempty"`

	// ProducerDestinations: Monitoring configurations for sending metrics
	// to the producer project.
	// There can be multiple producer destinations, each one must have
	// a
	// different monitored resource type. A metric can be used in at
	// most
	// one producer destination.
	ProducerDestinations []*MonitoringDestination `json:"producerDestinations,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ConsumerDestinations") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConsumerDestinations") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Monitoring) MarshalJSON() ([]byte, error) {
	type noMethod Monitoring
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MonitoringDestination: Configuration of a specific monitoring
// destination (the producer project
// or the consumer project).
type MonitoringDestination struct {
	// Metrics: Names of the metrics to report to this monitoring
	// destination.
	// Each name must be defined in Service.metrics section.
	Metrics []string `json:"metrics,omitempty"`

	// MonitoredResource: The monitored resource type. The type must be
	// defined in
	// Service.monitored_resources section.
	MonitoredResource string `json:"monitoredResource,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metrics") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metrics") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MonitoringDestination) MarshalJSON() ([]byte, error) {
	type noMethod MonitoringDestination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OAuthRequirements: OAuth scopes are a way to define data and
// permissions on data. For example,
// there are scopes defined for "Read-only access to Google Calendar"
// and
// "Access to Cloud Platform". Users can consent to a scope for an
// application,
// giving it permission to access that data on their behalf.
//
// OAuth scope specifications should be fairly coarse grained; a user
// will need
// to see and understand the text description of what your scope
// means.
//
// In most cases: use one or at most two OAuth scopes for an entire
// family of
// products. If your product has multiple APIs, you should probably be
// sharing
// the OAuth scope across all of those APIs.
//
// When you need finer grained OAuth consent screens: talk with your
// product
// management about how developers will use them in practice.
//
// Please note that even though each of the canonical scopes is enough
// for a
// request to be accepted and passed to the backend, a request can still
// fail
// due to the backend requiring additional scopes or permissions.
type OAuthRequirements struct {
	// CanonicalScopes: The list of publicly documented OAuth scopes that
	// are allowed access. An
	// OAuth token containing any of these scopes will be
	// accepted.
	//
	// Example:
	//
	//      canonical_scopes: https://www.googleapis.com/auth/calendar,
	//                        https://www.googleapis.com/auth/calendar.read
	CanonicalScopes string `json:"canonicalScopes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CanonicalScopes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CanonicalScopes") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OAuthRequirements) MarshalJSON() ([]byte, error) {
	type noMethod OAuthRequirements
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: This resource represents a long-running operation that is
// the result of a
// network API call.
type Operation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress.
	// If `true`, the operation is completed, and either `error` or
	// `response` is
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

// OperationMetadata: The metadata associated with a long running
// operation resource.
type OperationMetadata struct {
	// ProgressPercentage: Percentage of completion of this operation,
	// ranging from 0 to 100.
	ProgressPercentage int64 `json:"progressPercentage,omitempty"`

	// ResourceNames: The full name of the resources that this operation is
	// directly
	// associated with.
	ResourceNames []string `json:"resourceNames,omitempty"`

	// StartTime: The start time of the operation.
	StartTime string `json:"startTime,omitempty"`

	// Steps: Detailed status information for each step. The order is
	// undetermined.
	Steps []*Step `json:"steps,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProgressPercentage")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProgressPercentage") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *OperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod OperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Option: A protocol buffer option, which can be attached to a message,
// field,
// enumeration, etc.
type Option struct {
	// Name: The option's name. For protobuf built-in options (options
	// defined in
	// descriptor.proto), this is the short name. For example,
	// "map_entry".
	// For custom options, it should be the fully-qualified name. For
	// example,
	// "google.api.http".
	Name string `json:"name,omitempty"`

	// Value: The option's value packed in an Any message. If the value is a
	// primitive,
	// the corresponding wrapper type defined in
	// google/protobuf/wrappers.proto
	// should be used. If the value is an enum, it should be stored as an
	// int32
	// value using the google.protobuf.Int32Value type.
	Value googleapi.RawMessage `json:"value,omitempty"`

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

func (s *Option) MarshalJSON() ([]byte, error) {
	type noMethod Option
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Page: Represents a documentation page. A page can contain subpages to
// represent
// nested documentation set structure.
type Page struct {
	// Content: The Markdown content of the page. You can use <code>&#40;==
	// include {path} ==&#41;</code>
	// to include content from a Markdown file.
	Content string `json:"content,omitempty"`

	// Name: The name of the page. It will be used as an identity of the
	// page to
	// generate URI of the page, text of the link to this page in
	// navigation,
	// etc. The full page name (start from the root page name to this
	// page
	// concatenated with `.`) can be used as reference to the page in
	// your
	// documentation. For example:
	// <pre><code>pages:
	// - name: Tutorial
	//   content: &#40;== include tutorial.md ==&#41;
	//   subpages:
	//   - name: Java
	//     content: &#40;== include tutorial_java.md
	// ==&#41;
	// </code></pre>
	// You can reference `Java` page using Markdown reference link
	// syntax:
	// `Java`.
	Name string `json:"name,omitempty"`

	// Subpages: Subpages of this page. The order of subpages specified here
	// will be
	// honored in the generated docset.
	Subpages []*Page `json:"subpages,omitempty"`

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

func (s *Page) MarshalJSON() ([]byte, error) {
	type noMethod Page
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
	// AuditConfigs: Specifies cloud audit logging configuration for this
	// policy.
	AuditConfigs []*AuditConfig `json:"auditConfigs,omitempty"`

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

	IamOwned bool `json:"iamOwned,omitempty"`

	// Version: Version of the `Policy`. The default version is 0.
	Version int64 `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AuditConfigs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuditConfigs") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Policy) MarshalJSON() ([]byte, error) {
	type noMethod Policy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Quota: Quota configuration helps to achieve fairness and budgeting in
// service
// usage.
//
// The quota configuration works this way:
// - The service configuration defines a set of metrics.
// - For API calls, the quota.metric_rules maps methods to metrics with
//   corresponding costs.
// - The quota.limits defines limits on the metrics, which will be used
// for
//   quota checks at runtime.
//
// An example quota configuration in yaml format:
//
//    quota:
//
//      - name: apiWriteQpsPerProject
//        metric: library.googleapis.com/write_calls
//        unit: "1/min/{project}"  # rate limit for consumer projects
//        values:
//          STANDARD: 10000
//
//
//      # The metric rules bind all methods to the read_calls metric,
//      # except for the UpdateBook and DeleteBook methods. These two
// methods
//      # are mapped to the write_calls metric, with the UpdateBook
// method
//      # consuming at twice rate as the DeleteBook method.
//      metric_rules:
//      - selector: "*"
//        metric_costs:
//          library.googleapis.com/read_calls: 1
//      - selector: google.example.library.v1.LibraryService.UpdateBook
//        metric_costs:
//          library.googleapis.com/write_calls: 2
//      - selector: google.example.library.v1.LibraryService.DeleteBook
//        metric_costs:
//          library.googleapis.com/write_calls: 1
//
//  Corresponding Metric definition:
//
//      metrics:
//      - name: library.googleapis.com/read_calls
//        display_name: Read requests
//        metric_kind: DELTA
//        value_type: INT64
//
//      - name: library.googleapis.com/write_calls
//        display_name: Write requests
//        metric_kind: DELTA
//        value_type: INT64
type Quota struct {
	// Limits: List of `QuotaLimit` definitions for the service.
	Limits []*QuotaLimit `json:"limits,omitempty"`

	// MetricRules: List of `MetricRule` definitions, each one mapping a
	// selected method to one
	// or more metrics.
	MetricRules []*MetricRule `json:"metricRules,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Limits") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Limits") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Quota) MarshalJSON() ([]byte, error) {
	type noMethod Quota
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QuotaLimit: `QuotaLimit` defines a specific limit that applies over a
// specified duration
// for a limit type. There can be at most one limit for a duration and
// limit
// type combination defined within a `QuotaGroup`.
type QuotaLimit struct {
	// DefaultLimit: Default number of tokens that can be consumed during
	// the specified
	// duration. This is the number of tokens assigned when a
	// client
	// application developer activates the service for his/her
	// project.
	//
	// Specifying a value of 0 will block all requests. This can be used if
	// you
	// are provisioning quota to selected consumers and blocking
	// others.
	// Similarly, a value of -1 will indicate an unlimited quota. No
	// other
	// negative values are allowed.
	//
	// Used by group-based quotas only.
	DefaultLimit int64 `json:"defaultLimit,omitempty,string"`

	// Description: Optional. User-visible, extended description for this
	// quota limit.
	// Should be used only when more context is needed to understand this
	// limit
	// than provided by the limit's display name (see: `display_name`).
	Description string `json:"description,omitempty"`

	// DisplayName: User-visible display name for this limit.
	// Optional. If not set, the UI will provide a default display name
	// based on
	// the quota configuration. This field can be used to override the
	// default
	// display name generated from the configuration.
	DisplayName string `json:"displayName,omitempty"`

	// Duration: Duration of this limit in textual notation. Example:
	// "100s", "24h", "1d".
	// For duration longer than a day, only multiple of days is supported.
	// We
	// support only "100s" and "1d" for now. Additional support will be
	// added in
	// the future. "0" indicates indefinite duration.
	//
	// Used by group-based quotas only.
	Duration string `json:"duration,omitempty"`

	// FreeTier: Free tier value displayed in the Developers Console for
	// this limit.
	// The free tier is the number of tokens that will be subtracted from
	// the
	// billed amount when billing is enabled.
	// This field can only be set on a limit with duration "1d", in a
	// billable
	// group; it is invalid on any other limit. If this field is not set,
	// it
	// defaults to 0, indicating that there is no free tier for this
	// service.
	//
	// Used by group-based quotas only.
	FreeTier int64 `json:"freeTier,omitempty,string"`

	// MaxLimit: Maximum number of tokens that can be consumed during the
	// specified
	// duration. Client application developers can override the default
	// limit up
	// to this maximum. If specified, this value cannot be set to a value
	// less
	// than the default limit. If not specified, it is set to the default
	// limit.
	//
	// To allow clients to apply overrides with no upper bound, set this to
	// -1,
	// indicating unlimited maximum quota.
	//
	// Used by group-based quotas only.
	MaxLimit int64 `json:"maxLimit,omitempty,string"`

	// Metric: The name of the metric this quota limit applies to. The quota
	// limits with
	// the same metric will be checked together during runtime. The metric
	// must be
	// defined within the service config.
	//
	// Used by metric-based quotas only.
	Metric string `json:"metric,omitempty"`

	// Name: Name of the quota limit. The name is used to refer to the limit
	// when
	// overriding the default limit on per-consumer basis.
	//
	// For metric-based quota limits, the name must be provided, and it must
	// be
	// unique within the service. The name can only include
	// alphanumeric
	// characters as well as '-'.
	//
	// The maximum length of the limit name is 64 characters.
	//
	// The name of a limit is used as a unique identifier for this
	// limit.
	// Therefore, once a limit has been put into use, its name should
	// be
	// immutable. You can use the display_name field to provide a
	// user-friendly
	// name for the limit. The display name can be evolved over time
	// without
	// affecting the identity of the limit.
	Name string `json:"name,omitempty"`

	// Unit: Specify the unit of the quota limit. It uses the same syntax
	// as
	// Metric.unit. The supported unit kinds are determined by the
	// quota
	// backend system.
	//
	// The [Google Service
	// Control](https://cloud.google.com/service-control)
	// supports the following unit components:
	// * One of the time intevals:
	//   * "/min"  for quota every minute.
	//   * "/d"  for quota every 24 hours, starting 00:00 US Pacific Time.
	//   * Otherwise the quota won't be reset by time, such as storage
	// limit.
	// * One and only one of the granted containers:
	//   * "/{project}" quota for a project
	//
	// Here are some examples:
	// * "1/min/{project}" for quota per minute per project.
	//
	// Note: the order of unit components is insignificant.
	// The "1" at the beginning is required to follow the metric unit
	// syntax.
	//
	// Used by metric-based quotas only.
	Unit string `json:"unit,omitempty"`

	// Values: Tiered limit values, currently only STANDARD is supported.
	Values map[string]string `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DefaultLimit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DefaultLimit") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QuotaLimit) MarshalJSON() ([]byte, error) {
	type noMethod QuotaLimit
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Rollout: A rollout resource that defines how service configuration
// versions are pushed
// to control plane systems. Typically, you create a new version of
// the
// service config, and then create a Rollout to push the service config.
type Rollout struct {
	// CreateTime: Creation time of the rollout. Readonly.
	CreateTime string `json:"createTime,omitempty"`

	// CreatedBy: The user who created the Rollout. Readonly.
	CreatedBy string `json:"createdBy,omitempty"`

	// DeleteServiceStrategy: The strategy associated with a rollout to
	// delete a `ManagedService`.
	// Readonly.
	DeleteServiceStrategy *DeleteServiceStrategy `json:"deleteServiceStrategy,omitempty"`

	// RolloutId: Optional unique identifier of this Rollout. Only lower
	// case letters, digits
	//  and '-' are allowed.
	//
	// If not specified by client, the server will generate one. The
	// generated id
	// will have the form of <date><revision number>, where "date" is the
	// create
	// date in ISO 8601 format.  "revision number" is a monotonically
	// increasing
	// positive number that is reset every day for each service.
	// An example of the generated rollout_id is '2016-02-16r1'
	RolloutId string `json:"rolloutId,omitempty"`

	// ServiceName: The name of the service associated with this Rollout.
	ServiceName string `json:"serviceName,omitempty"`

	// Status: The status of this rollout. Readonly. In case of a failed
	// rollout,
	// the system will automatically rollback to the current
	// Rollout
	// version. Readonly.
	//
	// Possible values:
	//   "ROLLOUT_STATUS_UNSPECIFIED" - No status specified.
	//   "IN_PROGRESS" - The Rollout is in progress.
	//   "SUCCESS" - The Rollout has completed successfully.
	//   "CANCELLED" - The Rollout has been cancelled. This can happen if
	// you have overlapping
	// Rollout pushes, and the previous ones will be cancelled.
	//   "FAILED" - The Rollout has failed and the rollback attempt has
	// failed too.
	//   "PENDING" - The Rollout has not started yet and is pending for
	// execution.
	//   "FAILED_ROLLED_BACK" - The Rollout has failed and rolled back to
	// the previous successful
	// Rollout.
	Status string `json:"status,omitempty"`

	// TrafficPercentStrategy: Google Service Control selects service
	// configurations based on
	// traffic percentage.
	TrafficPercentStrategy *TrafficPercentStrategy `json:"trafficPercentStrategy,omitempty"`

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

func (s *Rollout) MarshalJSON() ([]byte, error) {
	type noMethod Rollout
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Service: `Service` is the root object of Google service configuration
// schema. It
// describes basic information about a service, such as the name and
// the
// title, and delegates other aspects to sub-sections. Each sub-section
// is
// either a proto message or a repeated proto message that configures
// a
// specific aspect, such as auth. See each proto message definition for
// details.
//
// Example:
//
//     type: google.api.Service
//     config_version: 3
//     name: calendar.googleapis.com
//     title: Google Calendar API
//     apis:
//     - name: google.calendar.v3.Calendar
//     authentication:
//       providers:
//       - id: google_calendar_auth
//         jwks_uri: https://www.googleapis.com/oauth2/v1/certs
//         issuer: https://securetoken.google.com
//       rules:
//       - selector: "*"
//         requirements:
//           provider_id: google_calendar_auth
type Service struct {
	// Apis: A list of API interfaces exported by this service. Only the
	// `name` field
	// of the google.protobuf.Api needs to be provided by the
	// configuration
	// author, as the remaining fields will be derived from the IDL during
	// the
	// normalization process. It is an error to specify an API interface
	// here
	// which cannot be resolved against the associated IDL files.
	Apis []*Api `json:"apis,omitempty"`

	// Authentication: Auth configuration.
	Authentication *Authentication `json:"authentication,omitempty"`

	// Backend: API backend configuration.
	Backend *Backend `json:"backend,omitempty"`

	// ConfigVersion: The semantic version of the service configuration. The
	// config version
	// affects the interpretation of the service configuration. For
	// example,
	// certain features are enabled by default for certain config
	// versions.
	// The latest config version is `3`.
	ConfigVersion int64 `json:"configVersion,omitempty"`

	// Context: Context configuration.
	Context *Context `json:"context,omitempty"`

	// Control: Configuration for the service control plane.
	Control *Control `json:"control,omitempty"`

	// CustomError: Custom error configuration.
	CustomError *CustomError `json:"customError,omitempty"`

	// Documentation: Additional API documentation.
	Documentation *Documentation `json:"documentation,omitempty"`

	// Endpoints: Configuration for network endpoints.  If this is empty,
	// then an endpoint
	// with the same name as the service is automatically generated to
	// service all
	// defined APIs.
	Endpoints []*Endpoint `json:"endpoints,omitempty"`

	// Enums: A list of all enum types included in this API service.
	// Enums
	// referenced directly or indirectly by the `apis` are
	// automatically
	// included.  Enums which are not referenced but shall be
	// included
	// should be listed here by name. Example:
	//
	//     enums:
	//     - name: google.someapi.v1.SomeEnum
	Enums []*Enum `json:"enums,omitempty"`

	// Experimental: Experimental configuration.
	Experimental *Experimental `json:"experimental,omitempty"`

	// Http: HTTP configuration.
	Http *Http `json:"http,omitempty"`

	// Id: A unique ID for a specific instance of this message, typically
	// assigned
	// by the client for tracking purpose. If empty, the server may choose
	// to
	// generate one instead.
	Id string `json:"id,omitempty"`

	// Logging: Logging configuration.
	Logging *Logging `json:"logging,omitempty"`

	// Logs: Defines the logs used by this service.
	Logs []*LogDescriptor `json:"logs,omitempty"`

	// Metrics: Defines the metrics used by this service.
	Metrics []*MetricDescriptor `json:"metrics,omitempty"`

	// MonitoredResources: Defines the monitored resources used by this
	// service. This is required
	// by the Service.monitoring and Service.logging configurations.
	MonitoredResources []*MonitoredResourceDescriptor `json:"monitoredResources,omitempty"`

	// Monitoring: Monitoring configuration.
	Monitoring *Monitoring `json:"monitoring,omitempty"`

	// Name: The DNS address at which this service is available,
	// e.g. `calendar.googleapis.com`.
	Name string `json:"name,omitempty"`

	// ProducerProjectId: The Google project that owns this service.
	ProducerProjectId string `json:"producerProjectId,omitempty"`

	// Quota: Quota configuration.
	Quota *Quota `json:"quota,omitempty"`

	// SourceInfo: Output only. The source information for this
	// configuration if available.
	SourceInfo *SourceInfo `json:"sourceInfo,omitempty"`

	// SystemParameters: System parameter configuration.
	SystemParameters *SystemParameters `json:"systemParameters,omitempty"`

	// SystemTypes: A list of all proto message types included in this API
	// service.
	// It serves similar purpose as [google.api.Service.types], except
	// that
	// these types are not needed by user-defined APIs. Therefore, they will
	// not
	// show up in the generated discovery doc. This field should only be
	// used
	// to define system APIs in ESF.
	SystemTypes []*Type `json:"systemTypes,omitempty"`

	// Title: The product title for this service.
	Title string `json:"title,omitempty"`

	// Types: A list of all proto message types included in this API
	// service.
	// Types referenced directly or indirectly by the `apis`
	// are
	// automatically included.  Messages which are not referenced but
	// shall be included, such as types used by the `google.protobuf.Any`
	// type,
	// should be listed here by name. Example:
	//
	//     types:
	//     - name: google.protobuf.Int32
	Types []*Type `json:"types,omitempty"`

	// Usage: Configuration controlling usage of this service.
	Usage *Usage `json:"usage,omitempty"`

	// Visibility: API visibility configuration.
	Visibility *Visibility `json:"visibility,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Apis") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Apis") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Service) MarshalJSON() ([]byte, error) {
	type noMethod Service
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

	// UpdateMask: OPTIONAL: A FieldMask specifying which fields of the
	// policy to modify. Only
	// the fields in the mask will be modified. If no mask is provided,
	// the
	// following default mask is used:
	// paths: "bindings, etag"
	// This field is only used by Cloud IAM.
	UpdateMask string `json:"updateMask,omitempty"`

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

// SourceContext: `SourceContext` represents information about the
// source of a
// protobuf element, like the file in which it is defined.
type SourceContext struct {
	// FileName: The path-qualified name of the .proto file that contained
	// the associated
	// protobuf element.  For example:
	// "google/protobuf/source_context.proto".
	FileName string `json:"fileName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FileName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FileName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SourceContext) MarshalJSON() ([]byte, error) {
	type noMethod SourceContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SourceInfo: Source information used to create a Service Config
type SourceInfo struct {
	// SourceFiles: All files used during config generation.
	SourceFiles []googleapi.RawMessage `json:"sourceFiles,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SourceFiles") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SourceFiles") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SourceInfo) MarshalJSON() ([]byte, error) {
	type noMethod SourceInfo
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

// Step: Represents the status of one operation step.
type Step struct {
	// Description: The short description of the step.
	Description string `json:"description,omitempty"`

	// Status: The status code.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED" - Unspecifed code.
	//   "DONE" - The operation or step has completed without errors.
	//   "NOT_STARTED" - The operation or step has not started yet.
	//   "IN_PROGRESS" - The operation or step is in progress.
	//   "FAILED" - The operation or step has completed with errors. If the
	// operation is
	// rollbackable, the rollback completed with errors too.
	//   "CANCELLED" - The operation or step has completed with
	// cancellation.
	Status string `json:"status,omitempty"`

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

func (s *Step) MarshalJSON() ([]byte, error) {
	type noMethod Step
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SubmitConfigSourceRequest: Request message for SubmitConfigSource
// method.
type SubmitConfigSourceRequest struct {
	// ConfigSource: The source configuration for the service.
	ConfigSource *ConfigSource `json:"configSource,omitempty"`

	// ValidateOnly: Optional. If set, this will result in the generation of
	// a
	// `google.api.Service` configuration based on the `ConfigSource`
	// provided,
	// but the generated config and the sources will NOT be persisted.
	ValidateOnly bool `json:"validateOnly,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConfigSource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConfigSource") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SubmitConfigSourceRequest) MarshalJSON() ([]byte, error) {
	type noMethod SubmitConfigSourceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SubmitConfigSourceResponse: Response message for SubmitConfigSource
// method.
type SubmitConfigSourceResponse struct {
	// ServiceConfig: The generated service configuration.
	ServiceConfig *Service `json:"serviceConfig,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ServiceConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ServiceConfig") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SubmitConfigSourceResponse) MarshalJSON() ([]byte, error) {
	type noMethod SubmitConfigSourceResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SystemParameter: Define a parameter's name and location. The
// parameter may be passed as either
// an HTTP header or a URL query parameter, and if both are passed the
// behavior
// is implementation-dependent.
type SystemParameter struct {
	// HttpHeader: Define the HTTP header name to use for the parameter. It
	// is case
	// insensitive.
	HttpHeader string `json:"httpHeader,omitempty"`

	// Name: Define the name of the parameter, such as "api_key" . It is
	// case sensitive.
	Name string `json:"name,omitempty"`

	// UrlQueryParameter: Define the URL query parameter name to use for the
	// parameter. It is case
	// sensitive.
	UrlQueryParameter string `json:"urlQueryParameter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HttpHeader") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HttpHeader") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SystemParameter) MarshalJSON() ([]byte, error) {
	type noMethod SystemParameter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SystemParameterRule: Define a system parameter rule mapping system
// parameter definitions to
// methods.
type SystemParameterRule struct {
	// Parameters: Define parameters. Multiple names may be defined for a
	// parameter.
	// For a given method call, only one of them should be used. If
	// multiple
	// names are used the behavior is implementation-dependent.
	// If none of the specified names are present the behavior
	// is
	// parameter-dependent.
	Parameters []*SystemParameter `json:"parameters,omitempty"`

	// Selector: Selects the methods to which this rule applies. Use '*' to
	// indicate all
	// methods in all APIs.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Parameters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Parameters") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SystemParameterRule) MarshalJSON() ([]byte, error) {
	type noMethod SystemParameterRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SystemParameters: ### System parameter configuration
//
// A system parameter is a special kind of parameter defined by the
// API
// system, not by an individual API. It is typically mapped to an HTTP
// header
// and/or a URL query parameter. This configuration specifies which
// methods
// change the names of the system parameters.
type SystemParameters struct {
	// Rules: Define system parameters.
	//
	// The parameters defined here will override the default
	// parameters
	// implemented by the system. If this field is missing from the
	// service
	// config, default system parameters will be used. Default system
	// parameters
	// and names is implementation-dependent.
	//
	// Example: define api key for all methods
	//
	//     system_parameters
	//       rules:
	//         - selector: "*"
	//           parameters:
	//             - name: api_key
	//               url_query_parameter: api_key
	//
	//
	// Example: define 2 api key names for a specific method.
	//
	//     system_parameters
	//       rules:
	//         - selector: "/ListShelves"
	//           parameters:
	//             - name: api_key
	//               http_header: Api-Key1
	//             - name: api_key
	//               http_header: Api-Key2
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*SystemParameterRule `json:"rules,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Rules") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Rules") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SystemParameters) MarshalJSON() ([]byte, error) {
	type noMethod SystemParameters
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestIamPermissionsRequest: Request message for `TestIamPermissions`
// method.
type TestIamPermissionsRequest struct {
	// Permissions: The set of permissions to check for the `resource`.
	// Permissions with
	// wildcards (such as '*' or 'storage.*') are not allowed. For
	// more
	// information see
	// [IAM
	// Overview](https://cloud.google.com/iam/docs/overview#permissions).
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

// TrafficPercentStrategy: Strategy that specifies how clients of Google
// Service Controller want to
// send traffic to use different config versions. This is generally
// used by API proxy to split traffic based on your configured
// precentage for
// each config version.
//
// One example of how to gradually rollout a new service configuration
// using
// this
// strategy:
// Day 1
//
//     Rollout {
//       id: "example.googleapis.com/rollout_20160206"
//       traffic_percent_strategy {
//         percentages: {
//           "example.googleapis.com/20160201": 70.00
//           "example.googleapis.com/20160206": 30.00
//         }
//       }
//     }
//
// Day 2
//
//     Rollout {
//       id: "example.googleapis.com/rollout_20160207"
//       traffic_percent_strategy: {
//         percentages: {
//           "example.googleapis.com/20160206": 100.00
//         }
//       }
//     }
type TrafficPercentStrategy struct {
	// Percentages: Maps service configuration IDs to their corresponding
	// traffic percentage.
	// Key is the service configuration ID, Value is the traffic
	// percentage
	// which must be greater than 0.0 and the sum must equal to 100.0.
	Percentages map[string]float64 `json:"percentages,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Percentages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Percentages") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TrafficPercentStrategy) MarshalJSON() ([]byte, error) {
	type noMethod TrafficPercentStrategy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Type: A protocol buffer message type.
type Type struct {
	// Fields: The list of fields.
	Fields []*Field `json:"fields,omitempty"`

	// Name: The fully qualified message name.
	Name string `json:"name,omitempty"`

	// Oneofs: The list of types appearing in `oneof` definitions in this
	// type.
	Oneofs []string `json:"oneofs,omitempty"`

	// Options: The protocol buffer options.
	Options []*Option `json:"options,omitempty"`

	// SourceContext: The source context.
	SourceContext *SourceContext `json:"sourceContext,omitempty"`

	// Syntax: The source syntax.
	//
	// Possible values:
	//   "SYNTAX_PROTO2" - Syntax `proto2`.
	//   "SYNTAX_PROTO3" - Syntax `proto3`.
	Syntax string `json:"syntax,omitempty"`

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

func (s *Type) MarshalJSON() ([]byte, error) {
	type noMethod Type
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UndeleteServiceResponse: Response message for UndeleteService method.
type UndeleteServiceResponse struct {
	// Service: Revived service resource.
	Service *ManagedService `json:"service,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Service") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Service") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UndeleteServiceResponse) MarshalJSON() ([]byte, error) {
	type noMethod UndeleteServiceResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Usage: Configuration controlling usage of a service.
type Usage struct {
	// ProducerNotificationChannel: The full resource name of a channel used
	// for sending notifications to the
	// service producer.
	//
	// Google Service Management currently only supports
	// [Google Cloud Pub/Sub](https://cloud.google.com/pubsub) as a
	// notification
	// channel. To use Google Cloud Pub/Sub as the channel, this must be the
	// name
	// of a Cloud Pub/Sub topic that uses the Cloud Pub/Sub topic name
	// format
	// documented in https://cloud.google.com/pubsub/docs/overview.
	ProducerNotificationChannel string `json:"producerNotificationChannel,omitempty"`

	// Requirements: Requirements that must be satisfied before a consumer
	// project can use the
	// service. Each requirement is of the form
	// <service.name>/<requirement-id>;
	// for example 'serviceusage.googleapis.com/billing-enabled'.
	Requirements []string `json:"requirements,omitempty"`

	// Rules: A list of usage rules that apply to individual API
	// methods.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*UsageRule `json:"rules,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ProducerNotificationChannel") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "ProducerNotificationChannel") to include in API requests with the
	// JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Usage) MarshalJSON() ([]byte, error) {
	type noMethod Usage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UsageRule: Usage configuration rules for the service.
//
// NOTE: Under development.
//
//
// Use this rule to configure unregistered calls for the service.
// Unregistered
// calls are calls that do not contain consumer project
// identity.
// (Example: calls that do not contain an API key).
// By default, API methods do not allow unregistered calls, and each
// method call
// must be identified by a consumer project identity. Use this rule
// to
// allow/disallow unregistered calls.
//
// Example of an API that wants to allow unregistered calls for entire
// service.
//
//     usage:
//       rules:
//       - selector: "*"
//         allow_unregistered_calls: true
//
// Example of a method that wants to allow unregistered calls.
//
//     usage:
//       rules:
//       - selector:
// "google.example.library.v1.LibraryService.CreateBook"
//         allow_unregistered_calls: true
type UsageRule struct {
	// AllowUnregisteredCalls: True, if the method allows unregistered
	// calls; false otherwise.
	AllowUnregisteredCalls bool `json:"allowUnregisteredCalls,omitempty"`

	// Selector: Selects the methods to which this rule applies. Use '*' to
	// indicate all
	// methods in all APIs.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// SkipServiceControl: True, if the method should skip service control.
	// If so, no control plane
	// feature (like quota and billing) will be enabled.
	SkipServiceControl bool `json:"skipServiceControl,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AllowUnregisteredCalls") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowUnregisteredCalls")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *UsageRule) MarshalJSON() ([]byte, error) {
	type noMethod UsageRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Visibility: `Visibility` defines restrictions for the visibility of
// service
// elements.  Restrictions are specified using visibility labels
// (e.g., TRUSTED_TESTER) that are elsewhere linked to users and
// projects.
//
// Users and projects can have access to more than one visibility label.
// The
// effective visibility for multiple labels is the union of each
// label's
// elements, plus any unrestricted elements.
//
// If an element and its parents have no restrictions, visibility
// is
// unconditionally granted.
//
// Example:
//
//     visibility:
//       rules:
//       - selector: google.calendar.Calendar.EnhancedSearch
//         restriction: TRUSTED_TESTER
//       - selector: google.calendar.Calendar.Delegate
//         restriction: GOOGLE_INTERNAL
//
// Here, all methods are publicly visible except for the restricted
// methods
// EnhancedSearch and Delegate.
type Visibility struct {
	// Rules: A list of visibility rules that apply to individual API
	// elements.
	//
	// **NOTE:** All service configuration rules follow "last one wins"
	// order.
	Rules []*VisibilityRule `json:"rules,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Rules") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Rules") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Visibility) MarshalJSON() ([]byte, error) {
	type noMethod Visibility
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// VisibilityRule: A visibility rule provides visibility configuration
// for an individual API
// element.
type VisibilityRule struct {
	// Restriction: A comma-separated list of visibility labels that apply
	// to the `selector`.
	// Any of the listed labels can be used to grant the visibility.
	//
	// If a rule has multiple labels, removing one of the labels but not all
	// of
	// them can break clients.
	//
	// Example:
	//
	//     visibility:
	//       rules:
	//       - selector: google.calendar.Calendar.EnhancedSearch
	//         restriction: GOOGLE_INTERNAL, TRUSTED_TESTER
	//
	// Removing GOOGLE_INTERNAL from this restriction will break clients
	// that
	// rely on this method and only had access to it through
	// GOOGLE_INTERNAL.
	Restriction string `json:"restriction,omitempty"`

	// Selector: Selects methods, messages, fields, enums, etc. to which
	// this rule applies.
	//
	// Refer to selector for syntax details.
	Selector string `json:"selector,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Restriction") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Restriction") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *VisibilityRule) MarshalJSON() ([]byte, error) {
	type noMethod VisibilityRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "servicemanagement.operations.get":

type OperationsGetCall struct {
	s            *APIService
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
func (r *OperationsService) Get(name string) *OperationsGetCall {
	c := &OperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsGetCall) Fields(s ...googleapi.Field) *OperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsGetCall) IfNoneMatch(entityTag string) *OperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsGetCall) Context(ctx context.Context) *OperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "servicemanagement.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *OperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "flatPath": "v1/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^operations/.+$",
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
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.operations.list":

type OperationsListCall struct {
	s            *APIService
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists service operations that match the specified filter in the
// request.
func (r *OperationsService) List() *OperationsListCall {
	c := &OperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Filter sets the optional parameter "filter": A string for filtering
// Operations.
//   The following filter fields are supported&#58;
//
//   * serviceName&#58; Required. Only `=` operator is allowed.
//   * startTime&#58; The time this job was started, in ISO 8601
// format.
//     Allowed operators are `>=`,  `>`, `<=`, and `<`.
//   * status&#58; Can be `done`, `in_progress`, or `failed`. Allowed
//     operators are `=`, and `!=`.
//
//   Filter expression supports conjunction (AND) and disjunction (OR)
//   logical operators. However, the serviceName restriction must be at
// the
//   top-level and can only be combined with other restrictions via the
// AND
//   logical operator.
//
//   Examples&#58;
//
//   * `serviceName={some-service}.googleapis.com`
//   * `serviceName={some-service}.googleapis.com AND
// startTime>="2017-02-01"
//   * `serviceName={some-service}.googleapis.com AND status=done`
//   * `serviceName={some-service}.googleapis.com AND (status=done OR
// startTime>="2017-02-01")`
func (c *OperationsListCall) Filter(filter string) *OperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// Name sets the optional parameter "name": Not used.
func (c *OperationsListCall) Name(name string) *OperationsListCall {
	c.urlParams_.Set("name", name)
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of operations to return. If unspecified, defaults to
// 50. The maximum value is 100.
func (c *OperationsListCall) PageSize(pageSize int64) *OperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *OperationsListCall) PageToken(pageToken string) *OperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsListCall) Fields(s ...googleapi.Field) *OperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsListCall) IfNoneMatch(entityTag string) *OperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsListCall) Context(ctx context.Context) *OperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
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
	//   "description": "Lists service operations that match the specified filter in the request.",
	//   "flatPath": "v1/operations",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.operations.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "filter": {
	//       "description": "A string for filtering Operations.\n  The following filter fields are supported\u0026#58;\n\n  * serviceName\u0026#58; Required. Only `=` operator is allowed.\n  * startTime\u0026#58; The time this job was started, in ISO 8601 format.\n    Allowed operators are `\u003e=`,  `\u003e`, `\u003c=`, and `\u003c`.\n  * status\u0026#58; Can be `done`, `in_progress`, or `failed`. Allowed\n    operators are `=`, and `!=`.\n\n  Filter expression supports conjunction (AND) and disjunction (OR)\n  logical operators. However, the serviceName restriction must be at the\n  top-level and can only be combined with other restrictions via the AND\n  logical operator.\n\n  Examples\u0026#58;\n\n  * `serviceName={some-service}.googleapis.com`\n  * `serviceName={some-service}.googleapis.com AND startTime\u003e=\"2017-02-01\"`\n  * `serviceName={some-service}.googleapis.com AND status=done`\n  * `serviceName={some-service}.googleapis.com AND (status=done OR startTime\u003e=\"2017-02-01\")`",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Not used.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The maximum number of operations to return. If unspecified, defaults to\n50. The maximum value is 100.",
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
	//   "path": "v1/operations",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *OperationsListCall) Pages(ctx context.Context, f func(*ListOperationsResponse) error) error {
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

// method id "servicemanagement.services.create":

type ServicesCreateCall struct {
	s              *APIService
	managedservice *ManagedService
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// Create: Creates a new managed service.
// Please note one producer project can own no more than 20
// services.
//
// Operation<response: ManagedService>
func (r *ServicesService) Create(managedservice *ManagedService) *ServicesCreateCall {
	c := &ServicesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.managedservice = managedservice
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesCreateCall) Fields(s ...googleapi.Field) *ServicesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesCreateCall) Context(ctx context.Context) *ServicesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.managedservice)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ServicesCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Creates a new managed service.\nPlease note one producer project can own no more than 20 services.\n\nOperation\u003cresponse: ManagedService\u003e",
	//   "flatPath": "v1/services",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/services",
	//   "request": {
	//     "$ref": "ManagedService"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.delete":

type ServicesDeleteCall struct {
	s           *APIService
	serviceName string
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Delete: Deletes a managed service. This method will change the
// service to the
// `Soft-Delete` state for 30 days. Within this period, service
// producers may
// call UndeleteService to restore the service.
// After 30 days, the service will be permanently
// deleted.
//
// Operation<response: google.protobuf.Empty>
func (r *ServicesService) Delete(serviceName string) *ServicesDeleteCall {
	c := &ServicesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesDeleteCall) Fields(s ...googleapi.Field) *ServicesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesDeleteCall) Context(ctx context.Context) *ServicesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ServicesDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Deletes a managed service. This method will change the service to the\n`Soft-Delete` state for 30 days. Within this period, service producers may\ncall UndeleteService to restore the service.\nAfter 30 days, the service will be permanently deleted.\n\nOperation\u003cresponse: google.protobuf.Empty\u003e",
	//   "flatPath": "v1/services/{serviceName}",
	//   "httpMethod": "DELETE",
	//   "id": "servicemanagement.services.delete",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.disable":

type ServicesDisableCall struct {
	s                     *APIService
	serviceName           string
	disableservicerequest *DisableServiceRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Disable: Disables a service for a project, so it can no longer be
// be used for the project. It prevents accidental usage that may
// cause
// unexpected billing charges or security leaks.
//
// Operation<response: DisableServiceResponse>
func (r *ServicesService) Disable(serviceName string, disableservicerequest *DisableServiceRequest) *ServicesDisableCall {
	c := &ServicesDisableCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.disableservicerequest = disableservicerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesDisableCall) Fields(s ...googleapi.Field) *ServicesDisableCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesDisableCall) Context(ctx context.Context) *ServicesDisableCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesDisableCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesDisableCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.disableservicerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:disable")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.disable" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ServicesDisableCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Disables a service for a project, so it can no longer be\nbe used for the project. It prevents accidental usage that may cause\nunexpected billing charges or security leaks.\n\nOperation\u003cresponse: DisableServiceResponse\u003e",
	//   "flatPath": "v1/services/{serviceName}:disable",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.disable",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "Name of the service to disable. Specifying an unknown service name\nwill cause the request to fail.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:disable",
	//   "request": {
	//     "$ref": "DisableServiceRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.enable":

type ServicesEnableCall struct {
	s                    *APIService
	serviceName          string
	enableservicerequest *EnableServiceRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Enable: Enables a service for a project, so it can be used
// for the project. See
// [Cloud Auth Guide](https://cloud.google.com/docs/authentication)
// for
// more information.
//
// Operation<response: EnableServiceResponse>
func (r *ServicesService) Enable(serviceName string, enableservicerequest *EnableServiceRequest) *ServicesEnableCall {
	c := &ServicesEnableCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.enableservicerequest = enableservicerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesEnableCall) Fields(s ...googleapi.Field) *ServicesEnableCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesEnableCall) Context(ctx context.Context) *ServicesEnableCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesEnableCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesEnableCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enableservicerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:enable")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.enable" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ServicesEnableCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Enables a service for a project, so it can be used\nfor the project. See\n[Cloud Auth Guide](https://cloud.google.com/docs/authentication) for\nmore information.\n\nOperation\u003cresponse: EnableServiceResponse\u003e",
	//   "flatPath": "v1/services/{serviceName}:enable",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.enable",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "Name of the service to enable. Specifying an unknown service name will\ncause the request to fail.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:enable",
	//   "request": {
	//     "$ref": "EnableServiceRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.generateConfigReport":

type ServicesGenerateConfigReportCall struct {
	s                           *APIService
	generateconfigreportrequest *GenerateConfigReportRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// GenerateConfigReport: Generates and returns a report (errors,
// warnings and changes from
// existing configurations) associated
// with
// GenerateConfigReportRequest.new_value
//
// If GenerateConfigReportRequest.old_value is
// specified,
// GenerateConfigReportRequest will contain a single ChangeReport based
// on the
// comparison between GenerateConfigReportRequest.new_value
// and
// GenerateConfigReportRequest.old_value.
// If GenerateConfigReportRequest.old_value is not specified, this
// method
// will compare GenerateConfigReportRequest.new_value with the last
// pushed
// service configuration.
func (r *ServicesService) GenerateConfigReport(generateconfigreportrequest *GenerateConfigReportRequest) *ServicesGenerateConfigReportCall {
	c := &ServicesGenerateConfigReportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.generateconfigreportrequest = generateconfigreportrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesGenerateConfigReportCall) Fields(s ...googleapi.Field) *ServicesGenerateConfigReportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesGenerateConfigReportCall) Context(ctx context.Context) *ServicesGenerateConfigReportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesGenerateConfigReportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesGenerateConfigReportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.generateconfigreportrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services:generateConfigReport")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.generateConfigReport" call.
// Exactly one of *GenerateConfigReportResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *GenerateConfigReportResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesGenerateConfigReportCall) Do(opts ...googleapi.CallOption) (*GenerateConfigReportResponse, error) {
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
	ret := &GenerateConfigReportResponse{
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
	//   "description": "Generates and returns a report (errors, warnings and changes from\nexisting configurations) associated with\nGenerateConfigReportRequest.new_value\n\nIf GenerateConfigReportRequest.old_value is specified,\nGenerateConfigReportRequest will contain a single ChangeReport based on the\ncomparison between GenerateConfigReportRequest.new_value and\nGenerateConfigReportRequest.old_value.\nIf GenerateConfigReportRequest.old_value is not specified, this method\nwill compare GenerateConfigReportRequest.new_value with the last pushed\nservice configuration.",
	//   "flatPath": "v1/services:generateConfigReport",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.generateConfigReport",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/services:generateConfigReport",
	//   "request": {
	//     "$ref": "GenerateConfigReportRequest"
	//   },
	//   "response": {
	//     "$ref": "GenerateConfigReportResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.get":

type ServicesGetCall struct {
	s            *APIService
	serviceName  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a managed service. Authentication is required unless the
// service is
// public.
func (r *ServicesService) Get(serviceName string) *ServicesGetCall {
	c := &ServicesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesGetCall) Fields(s ...googleapi.Field) *ServicesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServicesGetCall) IfNoneMatch(entityTag string) *ServicesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesGetCall) Context(ctx context.Context) *ServicesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.get" call.
// Exactly one of *ManagedService or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ManagedService.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesGetCall) Do(opts ...googleapi.CallOption) (*ManagedService, error) {
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
	ret := &ManagedService{
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
	//   "description": "Gets a managed service. Authentication is required unless the service is\npublic.",
	//   "flatPath": "v1/services/{serviceName}",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.services.get",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The name of the service.  See the `ServiceManager` overview for naming\nrequirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}",
	//   "response": {
	//     "$ref": "ManagedService"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.getConfig":

type ServicesGetConfigCall struct {
	s            *APIService
	serviceName  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetConfig: Gets a service configuration (version) for a managed
// service.
func (r *ServicesService) GetConfig(serviceName string) *ServicesGetConfigCall {
	c := &ServicesGetConfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	return c
}

// ConfigId sets the optional parameter "configId": The id of the
// service configuration resource.
func (c *ServicesGetConfigCall) ConfigId(configId string) *ServicesGetConfigCall {
	c.urlParams_.Set("configId", configId)
	return c
}

// View sets the optional parameter "view": Specifies which parts of the
// Service Config should be returned in the
// response.
//
// Possible values:
//   "BASIC"
//   "FULL"
func (c *ServicesGetConfigCall) View(view string) *ServicesGetConfigCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesGetConfigCall) Fields(s ...googleapi.Field) *ServicesGetConfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServicesGetConfigCall) IfNoneMatch(entityTag string) *ServicesGetConfigCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesGetConfigCall) Context(ctx context.Context) *ServicesGetConfigCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesGetConfigCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesGetConfigCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/config")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.getConfig" call.
// Exactly one of *Service or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Service.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesGetConfigCall) Do(opts ...googleapi.CallOption) (*Service, error) {
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
	ret := &Service{
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
	//   "description": "Gets a service configuration (version) for a managed service.",
	//   "flatPath": "v1/services/{serviceName}/config",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.services.getConfig",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "configId": {
	//       "description": "The id of the service configuration resource.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Specifies which parts of the Service Config should be returned in the\nresponse.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/config",
	//   "response": {
	//     "$ref": "Service"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.getIamPolicy":

type ServicesGetIamPolicyCall struct {
	s                   *APIService
	resource            string
	getiampolicyrequest *GetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// GetIamPolicy: Gets the access control policy for a resource.
// Returns an empty policy if the resource exists and does not have a
// policy
// set.
func (r *ServicesService) GetIamPolicy(resource string, getiampolicyrequest *GetIamPolicyRequest) *ServicesGetIamPolicyCall {
	c := &ServicesGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.getiampolicyrequest = getiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesGetIamPolicyCall) Fields(s ...googleapi.Field) *ServicesGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesGetIamPolicyCall) Context(ctx context.Context) *ServicesGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "servicemanagement.services.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Gets the access control policy for a resource.\nReturns an empty policy if the resource exists and does not have a policy\nset.",
	//   "flatPath": "v1/services/{servicesId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^services/[^/]+$",
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
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.list":

type ServicesListCall struct {
	s            *APIService
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists managed services.
//
// Returns all public services. For authenticated users, also returns
// all
// services the calling user has "servicemanagement.services.get"
// permission
// for.
//
// **BETA:** If the caller specifies the `consumer_id`, it returns only
// the
// services enabled on the consumer. The `consumer_id` must have the
// format
// of "project:{PROJECT-ID}".
func (r *ServicesService) List() *ServicesListCall {
	c := &ServicesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// ConsumerId sets the optional parameter "consumerId": Include services
// consumed by the specified consumer.
//
// The Google Service Management implementation accepts the
// following
// forms:
// - project:<project_id>
func (c *ServicesListCall) ConsumerId(consumerId string) *ServicesListCall {
	c.urlParams_.Set("consumerId", consumerId)
	return c
}

// PageSize sets the optional parameter "pageSize": Requested size of
// the next page of data.
func (c *ServicesListCall) PageSize(pageSize int64) *ServicesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Token identifying
// which result to start with; returned by a previous list
// call.
func (c *ServicesListCall) PageToken(pageToken string) *ServicesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ProducerProjectId sets the optional parameter "producerProjectId":
// Include services produced by the specified project.
func (c *ServicesListCall) ProducerProjectId(producerProjectId string) *ServicesListCall {
	c.urlParams_.Set("producerProjectId", producerProjectId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesListCall) Fields(s ...googleapi.Field) *ServicesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServicesListCall) IfNoneMatch(entityTag string) *ServicesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesListCall) Context(ctx context.Context) *ServicesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.list" call.
// Exactly one of *ListServicesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListServicesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesListCall) Do(opts ...googleapi.CallOption) (*ListServicesResponse, error) {
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
	ret := &ListServicesResponse{
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
	//   "description": "Lists managed services.\n\nReturns all public services. For authenticated users, also returns all\nservices the calling user has \"servicemanagement.services.get\" permission\nfor.\n\n**BETA:** If the caller specifies the `consumer_id`, it returns only the\nservices enabled on the consumer. The `consumer_id` must have the format\nof \"project:{PROJECT-ID}\".",
	//   "flatPath": "v1/services",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.services.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "consumerId": {
	//       "description": "Include services consumed by the specified consumer.\n\nThe Google Service Management implementation accepts the following\nforms:\n- project:\u003cproject_id\u003e",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested size of the next page of data.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Token identifying which result to start with; returned by a previous list\ncall.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "producerProjectId": {
	//       "description": "Include services produced by the specified project.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services",
	//   "response": {
	//     "$ref": "ListServicesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ServicesListCall) Pages(ctx context.Context, f func(*ListServicesResponse) error) error {
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

// method id "servicemanagement.services.setIamPolicy":

type ServicesSetIamPolicyCall struct {
	s                   *APIService
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy on the specified
// resource. Replaces any
// existing policy.
func (r *ServicesService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ServicesSetIamPolicyCall {
	c := &ServicesSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesSetIamPolicyCall) Fields(s ...googleapi.Field) *ServicesSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesSetIamPolicyCall) Context(ctx context.Context) *ServicesSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "servicemanagement.services.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Sets the access control policy on the specified resource. Replaces any\nexisting policy.",
	//   "flatPath": "v1/services/{servicesId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being specified.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^services/[^/]+$",
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
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.testIamPermissions":

type ServicesTestIamPermissionsCall struct {
	s                         *APIService
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
// If the resource does not exist, this will return an empty set
// of
// permissions, not a NOT_FOUND error.
//
// Note: This operation is designed to be used for building
// permission-aware
// UIs and command-line tools, not for authorization checking. This
// operation
// may "fail open" without warning.
func (r *ServicesService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ServicesTestIamPermissionsCall {
	c := &ServicesTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesTestIamPermissionsCall) Fields(s ...googleapi.Field) *ServicesTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesTestIamPermissionsCall) Context(ctx context.Context) *ServicesTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "servicemanagement.services.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	//   "description": "Returns permissions that a caller has on the specified resource.\nIf the resource does not exist, this will return an empty set of\npermissions, not a NOT_FOUND error.\n\nNote: This operation is designed to be used for building permission-aware\nUIs and command-line tools, not for authorization checking. This operation\nmay \"fail open\" without warning.",
	//   "flatPath": "v1/services/{servicesId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy detail is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^services/[^/]+$",
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
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.undelete":

type ServicesUndeleteCall struct {
	s           *APIService
	serviceName string
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Undelete: Revives a previously deleted managed service. The method
// restores the
// service using the configuration at the time the service was
// deleted.
// The target service must exist and must have been deleted within
// the
// last 30 days.
//
// Operation<response: UndeleteServiceResponse>
func (r *ServicesService) Undelete(serviceName string) *ServicesUndeleteCall {
	c := &ServicesUndeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesUndeleteCall) Fields(s ...googleapi.Field) *ServicesUndeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesUndeleteCall) Context(ctx context.Context) *ServicesUndeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesUndeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesUndeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:undelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.undelete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ServicesUndeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Revives a previously deleted managed service. The method restores the\nservice using the configuration at the time the service was deleted.\nThe target service must exist and must have been deleted within the\nlast 30 days.\n\nOperation\u003cresponse: UndeleteServiceResponse\u003e",
	//   "flatPath": "v1/services/{serviceName}:undelete",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.undelete",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The name of the service. See the [overview](/service-management/overview)\nfor naming requirements. For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:undelete",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.configs.create":

type ServicesConfigsCreateCall struct {
	s           *APIService
	serviceName string
	service     *Service
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Create: Creates a new service configuration (version) for a managed
// service.
// This method only stores the service configuration. To roll out the
// service
// configuration to backend systems please call
// CreateServiceRollout.
func (r *ServicesConfigsService) Create(serviceName string, service *Service) *ServicesConfigsCreateCall {
	c := &ServicesConfigsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.service = service
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesConfigsCreateCall) Fields(s ...googleapi.Field) *ServicesConfigsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesConfigsCreateCall) Context(ctx context.Context) *ServicesConfigsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesConfigsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesConfigsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.service)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/configs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.configs.create" call.
// Exactly one of *Service or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Service.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesConfigsCreateCall) Do(opts ...googleapi.CallOption) (*Service, error) {
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
	ret := &Service{
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
	//   "description": "Creates a new service configuration (version) for a managed service.\nThis method only stores the service configuration. To roll out the service\nconfiguration to backend systems please call\nCreateServiceRollout.",
	//   "flatPath": "v1/services/{serviceName}/configs",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.configs.create",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/configs",
	//   "request": {
	//     "$ref": "Service"
	//   },
	//   "response": {
	//     "$ref": "Service"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.configs.get":

type ServicesConfigsGetCall struct {
	s            *APIService
	serviceName  string
	configId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a service configuration (version) for a managed service.
func (r *ServicesConfigsService) Get(serviceName string, configId string) *ServicesConfigsGetCall {
	c := &ServicesConfigsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.configId = configId
	return c
}

// View sets the optional parameter "view": Specifies which parts of the
// Service Config should be returned in the
// response.
//
// Possible values:
//   "BASIC"
//   "FULL"
func (c *ServicesConfigsGetCall) View(view string) *ServicesConfigsGetCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesConfigsGetCall) Fields(s ...googleapi.Field) *ServicesConfigsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServicesConfigsGetCall) IfNoneMatch(entityTag string) *ServicesConfigsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesConfigsGetCall) Context(ctx context.Context) *ServicesConfigsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesConfigsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesConfigsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/configs/{configId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
		"configId":    c.configId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.configs.get" call.
// Exactly one of *Service or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Service.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesConfigsGetCall) Do(opts ...googleapi.CallOption) (*Service, error) {
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
	ret := &Service{
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
	//   "description": "Gets a service configuration (version) for a managed service.",
	//   "flatPath": "v1/services/{serviceName}/configs/{configId}",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.services.configs.get",
	//   "parameterOrder": [
	//     "serviceName",
	//     "configId"
	//   ],
	//   "parameters": {
	//     "configId": {
	//       "description": "The id of the service configuration resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Specifies which parts of the Service Config should be returned in the\nresponse.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/configs/{configId}",
	//   "response": {
	//     "$ref": "Service"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.configs.list":

type ServicesConfigsListCall struct {
	s            *APIService
	serviceName  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the history of the service configuration for a managed
// service,
// from the newest to the oldest.
func (r *ServicesConfigsService) List(serviceName string) *ServicesConfigsListCall {
	c := &ServicesConfigsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	return c
}

// PageSize sets the optional parameter "pageSize": The max number of
// items to include in the response list.
func (c *ServicesConfigsListCall) PageSize(pageSize int64) *ServicesConfigsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token of the
// page to retrieve.
func (c *ServicesConfigsListCall) PageToken(pageToken string) *ServicesConfigsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesConfigsListCall) Fields(s ...googleapi.Field) *ServicesConfigsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServicesConfigsListCall) IfNoneMatch(entityTag string) *ServicesConfigsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesConfigsListCall) Context(ctx context.Context) *ServicesConfigsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesConfigsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesConfigsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/configs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.configs.list" call.
// Exactly one of *ListServiceConfigsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListServiceConfigsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesConfigsListCall) Do(opts ...googleapi.CallOption) (*ListServiceConfigsResponse, error) {
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
	ret := &ListServiceConfigsResponse{
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
	//   "description": "Lists the history of the service configuration for a managed service,\nfrom the newest to the oldest.",
	//   "flatPath": "v1/services/{serviceName}/configs",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.services.configs.list",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The max number of items to include in the response list.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token of the page to retrieve.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/configs",
	//   "response": {
	//     "$ref": "ListServiceConfigsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ServicesConfigsListCall) Pages(ctx context.Context, f func(*ListServiceConfigsResponse) error) error {
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

// method id "servicemanagement.services.configs.submit":

type ServicesConfigsSubmitCall struct {
	s                         *APIService
	serviceName               string
	submitconfigsourcerequest *SubmitConfigSourceRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// Submit: Creates a new service configuration (version) for a managed
// service based
// on
// user-supplied configuration source files (for example:
// OpenAPI
// Specification). This method stores the source configurations as well
// as the
// generated service configuration. To rollout the service configuration
// to
// other services,
// please call CreateServiceRollout.
//
// Operation<response: SubmitConfigSourceResponse>
func (r *ServicesConfigsService) Submit(serviceName string, submitconfigsourcerequest *SubmitConfigSourceRequest) *ServicesConfigsSubmitCall {
	c := &ServicesConfigsSubmitCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.submitconfigsourcerequest = submitconfigsourcerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesConfigsSubmitCall) Fields(s ...googleapi.Field) *ServicesConfigsSubmitCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesConfigsSubmitCall) Context(ctx context.Context) *ServicesConfigsSubmitCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesConfigsSubmitCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesConfigsSubmitCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.submitconfigsourcerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/configs:submit")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.configs.submit" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ServicesConfigsSubmitCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Creates a new service configuration (version) for a managed service based\non\nuser-supplied configuration source files (for example: OpenAPI\nSpecification). This method stores the source configurations as well as the\ngenerated service configuration. To rollout the service configuration to\nother services,\nplease call CreateServiceRollout.\n\nOperation\u003cresponse: SubmitConfigSourceResponse\u003e",
	//   "flatPath": "v1/services/{serviceName}/configs:submit",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.configs.submit",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/configs:submit",
	//   "request": {
	//     "$ref": "SubmitConfigSourceRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.consumers.getIamPolicy":

type ServicesConsumersGetIamPolicyCall struct {
	s                   *APIService
	resource            string
	getiampolicyrequest *GetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// GetIamPolicy: Gets the access control policy for a resource.
// Returns an empty policy if the resource exists and does not have a
// policy
// set.
func (r *ServicesConsumersService) GetIamPolicy(resource string, getiampolicyrequest *GetIamPolicyRequest) *ServicesConsumersGetIamPolicyCall {
	c := &ServicesConsumersGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.getiampolicyrequest = getiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesConsumersGetIamPolicyCall) Fields(s ...googleapi.Field) *ServicesConsumersGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesConsumersGetIamPolicyCall) Context(ctx context.Context) *ServicesConsumersGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesConsumersGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesConsumersGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "servicemanagement.services.consumers.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesConsumersGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Gets the access control policy for a resource.\nReturns an empty policy if the resource exists and does not have a policy\nset.",
	//   "flatPath": "v1/services/{servicesId}/consumers/{consumersId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.consumers.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^services/[^/]+/consumers/[^/]+$",
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
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.consumers.setIamPolicy":

type ServicesConsumersSetIamPolicyCall struct {
	s                   *APIService
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy on the specified
// resource. Replaces any
// existing policy.
func (r *ServicesConsumersService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ServicesConsumersSetIamPolicyCall {
	c := &ServicesConsumersSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesConsumersSetIamPolicyCall) Fields(s ...googleapi.Field) *ServicesConsumersSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesConsumersSetIamPolicyCall) Context(ctx context.Context) *ServicesConsumersSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesConsumersSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesConsumersSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "servicemanagement.services.consumers.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesConsumersSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Sets the access control policy on the specified resource. Replaces any\nexisting policy.",
	//   "flatPath": "v1/services/{servicesId}/consumers/{consumersId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.consumers.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being specified.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^services/[^/]+/consumers/[^/]+$",
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
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.consumers.testIamPermissions":

type ServicesConsumersTestIamPermissionsCall struct {
	s                         *APIService
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
// If the resource does not exist, this will return an empty set
// of
// permissions, not a NOT_FOUND error.
//
// Note: This operation is designed to be used for building
// permission-aware
// UIs and command-line tools, not for authorization checking. This
// operation
// may "fail open" without warning.
func (r *ServicesConsumersService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ServicesConsumersTestIamPermissionsCall {
	c := &ServicesConsumersTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesConsumersTestIamPermissionsCall) Fields(s ...googleapi.Field) *ServicesConsumersTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesConsumersTestIamPermissionsCall) Context(ctx context.Context) *ServicesConsumersTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesConsumersTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesConsumersTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "servicemanagement.services.consumers.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesConsumersTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	//   "description": "Returns permissions that a caller has on the specified resource.\nIf the resource does not exist, this will return an empty set of\npermissions, not a NOT_FOUND error.\n\nNote: This operation is designed to be used for building permission-aware\nUIs and command-line tools, not for authorization checking. This operation\nmay \"fail open\" without warning.",
	//   "flatPath": "v1/services/{servicesId}/consumers/{consumersId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.consumers.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy detail is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^services/[^/]+/consumers/[^/]+$",
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
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.rollouts.create":

type ServicesRolloutsCreateCall struct {
	s           *APIService
	serviceName string
	rollout     *Rollout
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Create: Creates a new service configuration rollout. Based on
// rollout, the
// Google Service Management will roll out the service configurations
// to
// different backend services. For example, the logging configuration
// will be
// pushed to Google Cloud Logging.
//
// Please note that any previous pending and running Rollouts and
// associated
// Operations will be automatically cancelled so that the latest Rollout
// will
// not be blocked by previous Rollouts.
//
// Operation<response: Rollout>
func (r *ServicesRolloutsService) Create(serviceName string, rollout *Rollout) *ServicesRolloutsCreateCall {
	c := &ServicesRolloutsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.rollout = rollout
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesRolloutsCreateCall) Fields(s ...googleapi.Field) *ServicesRolloutsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesRolloutsCreateCall) Context(ctx context.Context) *ServicesRolloutsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesRolloutsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesRolloutsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.rollout)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/rollouts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.rollouts.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ServicesRolloutsCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Creates a new service configuration rollout. Based on rollout, the\nGoogle Service Management will roll out the service configurations to\ndifferent backend services. For example, the logging configuration will be\npushed to Google Cloud Logging.\n\nPlease note that any previous pending and running Rollouts and associated\nOperations will be automatically cancelled so that the latest Rollout will\nnot be blocked by previous Rollouts.\n\nOperation\u003cresponse: Rollout\u003e",
	//   "flatPath": "v1/services/{serviceName}/rollouts",
	//   "httpMethod": "POST",
	//   "id": "servicemanagement.services.rollouts.create",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/rollouts",
	//   "request": {
	//     "$ref": "Rollout"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/service.management"
	//   ]
	// }

}

// method id "servicemanagement.services.rollouts.get":

type ServicesRolloutsGetCall struct {
	s            *APIService
	serviceName  string
	rolloutId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a service configuration rollout.
func (r *ServicesRolloutsService) Get(serviceName string, rolloutId string) *ServicesRolloutsGetCall {
	c := &ServicesRolloutsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.rolloutId = rolloutId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesRolloutsGetCall) Fields(s ...googleapi.Field) *ServicesRolloutsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServicesRolloutsGetCall) IfNoneMatch(entityTag string) *ServicesRolloutsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesRolloutsGetCall) Context(ctx context.Context) *ServicesRolloutsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesRolloutsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesRolloutsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/rollouts/{rolloutId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
		"rolloutId":   c.rolloutId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.rollouts.get" call.
// Exactly one of *Rollout or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Rollout.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ServicesRolloutsGetCall) Do(opts ...googleapi.CallOption) (*Rollout, error) {
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
	ret := &Rollout{
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
	//   "description": "Gets a service configuration rollout.",
	//   "flatPath": "v1/services/{serviceName}/rollouts/{rolloutId}",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.services.rollouts.get",
	//   "parameterOrder": [
	//     "serviceName",
	//     "rolloutId"
	//   ],
	//   "parameters": {
	//     "rolloutId": {
	//       "description": "The id of the rollout resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/rollouts/{rolloutId}",
	//   "response": {
	//     "$ref": "Rollout"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// method id "servicemanagement.services.rollouts.list":

type ServicesRolloutsListCall struct {
	s            *APIService
	serviceName  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the history of the service configuration rollouts for a
// managed
// service, from the newest to the oldest.
func (r *ServicesRolloutsService) List(serviceName string) *ServicesRolloutsListCall {
	c := &ServicesRolloutsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	return c
}

// Filter sets the optional parameter "filter": Use `filter` to return
// subset of rollouts.
// The following filters are supported:
//   -- To limit the results to only those in
//      [status](google.api.servicemanagement.v1.RolloutStatus)
// 'SUCCESS',
//      use filter='status=SUCCESS'
//   -- To limit the results to those in
//      [status](google.api.servicemanagement.v1.RolloutStatus)
// 'CANCELLED'
//      or 'FAILED', use filter='status=CANCELLED OR status=FAILED'
func (c *ServicesRolloutsListCall) Filter(filter string) *ServicesRolloutsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The max number of
// items to include in the response list.
func (c *ServicesRolloutsListCall) PageSize(pageSize int64) *ServicesRolloutsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token of the
// page to retrieve.
func (c *ServicesRolloutsListCall) PageToken(pageToken string) *ServicesRolloutsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesRolloutsListCall) Fields(s ...googleapi.Field) *ServicesRolloutsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServicesRolloutsListCall) IfNoneMatch(entityTag string) *ServicesRolloutsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesRolloutsListCall) Context(ctx context.Context) *ServicesRolloutsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesRolloutsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesRolloutsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}/rollouts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicemanagement.services.rollouts.list" call.
// Exactly one of *ListServiceRolloutsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListServiceRolloutsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesRolloutsListCall) Do(opts ...googleapi.CallOption) (*ListServiceRolloutsResponse, error) {
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
	ret := &ListServiceRolloutsResponse{
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
	//   "description": "Lists the history of the service configuration rollouts for a managed\nservice, from the newest to the oldest.",
	//   "flatPath": "v1/services/{serviceName}/rollouts",
	//   "httpMethod": "GET",
	//   "id": "servicemanagement.services.rollouts.list",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Use `filter` to return subset of rollouts.\nThe following filters are supported:\n  -- To limit the results to only those in\n     [status](google.api.servicemanagement.v1.RolloutStatus) 'SUCCESS',\n     use filter='status=SUCCESS'\n  -- To limit the results to those in\n     [status](google.api.servicemanagement.v1.RolloutStatus) 'CANCELLED'\n     or 'FAILED', use filter='status=CANCELLED OR status=FAILED'",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The max number of items to include in the response list.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token of the page to retrieve.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "serviceName": {
	//       "description": "The name of the service.  See the [overview](/service-management/overview)\nfor naming requirements.  For example: `example.googleapis.com`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}/rollouts",
	//   "response": {
	//     "$ref": "ListServiceRolloutsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/service.management",
	//     "https://www.googleapis.com/auth/service.management.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ServicesRolloutsListCall) Pages(ctx context.Context, f func(*ListServiceRolloutsResponse) error) error {
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
