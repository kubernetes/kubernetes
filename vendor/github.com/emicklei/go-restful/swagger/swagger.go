// Package swagger implements the structures of the Swagger
// https://github.com/wordnik/swagger-spec/blob/master/versions/1.2.md
package swagger

const swaggerVersion = "1.2"

// 4.3.3 Data Type Fields
type DataTypeFields struct {
	Type         *string  `json:"type,omitempty"` // if Ref not used
	Ref          *string  `json:"$ref,omitempty"` // if Type not used
	Format       string   `json:"format,omitempty"`
	DefaultValue Special  `json:"defaultValue,omitempty"`
	Enum         []string `json:"enum,omitempty"`
	Minimum      string   `json:"minimum,omitempty"`
	Maximum      string   `json:"maximum,omitempty"`
	Items        *Item    `json:"items,omitempty"`
	UniqueItems  *bool    `json:"uniqueItems,omitempty"`
}

type Special string

// 4.3.4 Items Object
type Item struct {
	Type   *string `json:"type,omitempty"`
	Ref    *string `json:"$ref,omitempty"`
	Format string  `json:"format,omitempty"`
}

// 5.1 Resource Listing
type ResourceListing struct {
	SwaggerVersion string          `json:"swaggerVersion"` // e.g 1.2
	Apis           []Resource      `json:"apis"`
	ApiVersion     string          `json:"apiVersion"`
	Info           Info            `json:"info"`
	Authorizations []Authorization `json:"authorizations,omitempty"`
}

// 5.1.2 Resource Object
type Resource struct {
	Path        string `json:"path"` // relative or absolute, must start with /
	Description string `json:"description"`
}

// 5.1.3 Info Object
type Info struct {
	Title             string `json:"title"`
	Description       string `json:"description"`
	TermsOfServiceUrl string `json:"termsOfServiceUrl,omitempty"`
	Contact           string `json:"contact,omitempty"`
	License           string `json:"license,omitempty"`
	LicenseUrl        string `json:"licenseUrl,omitempty"`
}

// 5.1.5
type Authorization struct {
	Type       string      `json:"type"`
	PassAs     string      `json:"passAs"`
	Keyname    string      `json:"keyname"`
	Scopes     []Scope     `json:"scopes"`
	GrantTypes []GrantType `json:"grandTypes"`
}

// 5.1.6, 5.2.11
type Scope struct {
	// Required. The name of the scope.
	Scope string `json:"scope"`
	// Recommended. A short description of the scope.
	Description string `json:"description"`
}

// 5.1.7
type GrantType struct {
	Implicit          Implicit          `json:"implicit"`
	AuthorizationCode AuthorizationCode `json:"authorization_code"`
}

// 5.1.8 Implicit Object
type Implicit struct {
	// Required. The login endpoint definition.
	loginEndpoint LoginEndpoint `json:"loginEndpoint"`
	// An optional alternative name to standard "access_token" OAuth2 parameter.
	TokenName string `json:"tokenName"`
}

// 5.1.9 Authorization Code Object
type AuthorizationCode struct {
	TokenRequestEndpoint TokenRequestEndpoint `json:"tokenRequestEndpoint"`
	TokenEndpoint        TokenEndpoint        `json:"tokenEndpoint"`
}

// 5.1.10 Login Endpoint Object
type LoginEndpoint struct {
	// Required. The URL of the authorization endpoint for the implicit grant flow. The value SHOULD be in a URL format.
	Url string `json:"url"`
}

// 5.1.11 Token Request Endpoint Object
type TokenRequestEndpoint struct {
	// Required. The URL of the authorization endpoint for the authentication code grant flow. The value SHOULD be in a URL format.
	Url string `json:"url"`
	// An optional alternative name to standard "client_id" OAuth2 parameter.
	ClientIdName string `json:"clientIdName"`
	// An optional alternative name to the standard "client_secret" OAuth2 parameter.
	ClientSecretName string `json:"clientSecretName"`
}

// 5.1.12 Token Endpoint Object
type TokenEndpoint struct {
	// Required. The URL of the token endpoint for the authentication code grant flow. The value SHOULD be in a URL format.
	Url string `json:"url"`
	// An optional alternative name to standard "access_token" OAuth2 parameter.
	TokenName string `json:"tokenName"`
}

// 5.2 API Declaration
type ApiDeclaration struct {
	SwaggerVersion string          `json:"swaggerVersion"`
	ApiVersion     string          `json:"apiVersion"`
	BasePath       string          `json:"basePath"`
	ResourcePath   string          `json:"resourcePath"` // must start with /
	Apis           []Api           `json:"apis,omitempty"`
	Models         ModelList       `json:"models,omitempty"`
	Produces       []string        `json:"produces,omitempty"`
	Consumes       []string        `json:"consumes,omitempty"`
	Authorizations []Authorization `json:"authorizations,omitempty"`
}

// 5.2.2 API Object
type Api struct {
	Path        string      `json:"path"` // relative or absolute, must start with /
	Description string      `json:"description"`
	Operations  []Operation `json:"operations,omitempty"`
}

// 5.2.3 Operation Object
type Operation struct {
	DataTypeFields
	Method           string            `json:"method"`
	Summary          string            `json:"summary,omitempty"`
	Notes            string            `json:"notes,omitempty"`
	Nickname         string            `json:"nickname"`
	Authorizations   []Authorization   `json:"authorizations,omitempty"`
	Parameters       []Parameter       `json:"parameters"`
	ResponseMessages []ResponseMessage `json:"responseMessages,omitempty"` // optional
	Produces         []string          `json:"produces,omitempty"`
	Consumes         []string          `json:"consumes,omitempty"`
	Deprecated       string            `json:"deprecated,omitempty"`
}

// 5.2.4 Parameter Object
type Parameter struct {
	DataTypeFields
	ParamType     string `json:"paramType"` // path,query,body,header,form
	Name          string `json:"name"`
	Description   string `json:"description"`
	Required      bool   `json:"required"`
	AllowMultiple bool   `json:"allowMultiple"`
}

// 5.2.5 Response Message Object
type ResponseMessage struct {
	Code          int    `json:"code"`
	Message       string `json:"message"`
	ResponseModel string `json:"responseModel,omitempty"`
}

// 5.2.6, 5.2.7 Models Object
type Model struct {
	Id            string            `json:"id"`
	Description   string            `json:"description,omitempty"`
	Required      []string          `json:"required,omitempty"`
	Properties    ModelPropertyList `json:"properties"`
	SubTypes      []string          `json:"subTypes,omitempty"`
	Discriminator string            `json:"discriminator,omitempty"`
}

// 5.2.8 Properties Object
type ModelProperty struct {
	DataTypeFields
	Description string `json:"description,omitempty"`
}

// 5.2.10
type Authorizations map[string]Authorization
