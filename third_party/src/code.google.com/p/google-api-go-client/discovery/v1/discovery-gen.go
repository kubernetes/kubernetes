// Package discovery provides access to the APIs Discovery Service.
//
// See https://developers.google.com/discovery/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/discovery/v1"
//   ...
//   discoveryService, err := discovery.New(oauthHttpClient)
package discovery

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
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
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "discovery:v1"
const apiName = "discovery"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/discovery/v1/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Apis = NewApisService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Apis *ApisService
}

func NewApisService(s *Service) *ApisService {
	rs := &ApisService{s: s}
	return rs
}

type ApisService struct {
	s *Service
}

type DirectoryList struct {
	// DiscoveryVersion: Indicate the version of the Discovery API used to
	// generate this doc.
	DiscoveryVersion string `json:"discoveryVersion,omitempty"`

	// Items: The individual directory entries. One entry per api/version
	// pair.
	Items []*DirectoryListItems `json:"items,omitempty"`

	// Kind: The kind for this response.
	Kind string `json:"kind,omitempty"`
}

type DirectoryListItems struct {
	// Description: The description of this API.
	Description string `json:"description,omitempty"`

	// DiscoveryLink: A link to the discovery document.
	DiscoveryLink string `json:"discoveryLink,omitempty"`

	// DiscoveryRestUrl: The URL for the discovery REST document.
	DiscoveryRestUrl string `json:"discoveryRestUrl,omitempty"`

	// DocumentationLink: A link to human readable documentation for the
	// API.
	DocumentationLink string `json:"documentationLink,omitempty"`

	// Icons: Links to 16x16 and 32x32 icons representing the API.
	Icons *DirectoryListItemsIcons `json:"icons,omitempty"`

	// Id: The id of this API.
	Id string `json:"id,omitempty"`

	// Kind: The kind for this response.
	Kind string `json:"kind,omitempty"`

	// Labels: Labels for the status of this API, such as labs or
	// deprecated.
	Labels []string `json:"labels,omitempty"`

	// Name: The name of the API.
	Name string `json:"name,omitempty"`

	// Preferred: True if this version is the preferred version to use.
	Preferred bool `json:"preferred,omitempty"`

	// Title: The title of this API.
	Title string `json:"title,omitempty"`

	// Version: The version of the API.
	Version string `json:"version,omitempty"`
}

type DirectoryListItemsIcons struct {
	// X16: The URL of the 16x16 icon.
	X16 string `json:"x16,omitempty"`

	// X32: The URL of the 32x32 icon.
	X32 string `json:"x32,omitempty"`
}

type JsonSchema struct {
	// Ref: A reference to another schema. The value of this property is the
	// "id" of another schema.
	Ref string `json:"$ref,omitempty"`

	// AdditionalProperties: If this is a schema for an object, this
	// property is the schema for any additional properties with dynamic
	// keys on this object.
	AdditionalProperties *JsonSchema `json:"additionalProperties,omitempty"`

	// Annotations: Additional information about this property.
	Annotations *JsonSchemaAnnotations `json:"annotations,omitempty"`

	// Default: The default value of this property (if one exists).
	Default string `json:"default,omitempty"`

	// Description: A description of this object.
	Description string `json:"description,omitempty"`

	// Enum: Values this parameter may take (if it is an enum).
	Enum []string `json:"enum,omitempty"`

	// EnumDescriptions: The descriptions for the enums. Each position maps
	// to the corresponding value in the "enum" array.
	EnumDescriptions []string `json:"enumDescriptions,omitempty"`

	// Format: An additional regular expression or key that helps constrain
	// the value. For more details see:
	// http://tools.ietf.org/html/draft-zyp-json-schema-03#section-5.23
	Format string `json:"format,omitempty"`

	// Id: Unique identifier for this schema.
	Id string `json:"id,omitempty"`

	// Items: If this is a schema for an array, this property is the schema
	// for each element in the array.
	Items *JsonSchema `json:"items,omitempty"`

	// Location: Whether this parameter goes in the query or the path for
	// REST requests.
	Location string `json:"location,omitempty"`

	// Maximum: The maximum value of this parameter.
	Maximum string `json:"maximum,omitempty"`

	// Minimum: The minimum value of this parameter.
	Minimum string `json:"minimum,omitempty"`

	// Pattern: The regular expression this parameter must conform to. Uses
	// Java 6 regex format:
	// http://docs.oracle.com/javase/6/docs/api/java/util/regex/Pattern.html
	Pattern string `json:"pattern,omitempty"`

	// Properties: If this is a schema for an object, list the schema for
	// each property of this object.
	Properties *JsonSchemaProperties `json:"properties,omitempty"`

	// ReadOnly: The value is read-only, generated by the service. The value
	// cannot be modified by the client. If the value is included in a POST,
	// PUT, or PATCH request, it is ignored by the service.
	ReadOnly bool `json:"readOnly,omitempty"`

	// Repeated: Whether this parameter may appear multiple times.
	Repeated bool `json:"repeated,omitempty"`

	// Required: Whether the parameter is required.
	Required bool `json:"required,omitempty"`

	// Type: The value type for this schema. A list of values can be found
	// here: http://tools.ietf.org/html/draft-zyp-json-schema-03#section-5.1
	Type string `json:"type,omitempty"`

	// Variant: In a variant data type, the value of one property is used to
	// determine how to interpret the entire entity. Its value must exist in
	// a map of descriminant values to schema names.
	Variant *JsonSchemaVariant `json:"variant,omitempty"`
}

type JsonSchemaAnnotations struct {
	// Required: A list of methods for which this property is required on
	// requests.
	Required []string `json:"required,omitempty"`
}

type JsonSchemaProperties struct {
}

type JsonSchemaVariant struct {
	// Discriminant: The name of the type discriminant property.
	Discriminant string `json:"discriminant,omitempty"`

	// Map: The map of discriminant value to schema to use for parsing..
	Map []*JsonSchemaVariantMap `json:"map,omitempty"`
}

type JsonSchemaVariantMap struct {
	Ref string `json:"$ref,omitempty"`

	Type_value string `json:"type_value,omitempty"`
}

type RestDescription struct {
	// Auth: Authentication information.
	Auth *RestDescriptionAuth `json:"auth,omitempty"`

	// BasePath: [DEPRECATED] The base path for REST requests.
	BasePath string `json:"basePath,omitempty"`

	// BaseUrl: [DEPRECATED] The base URL for REST requests.
	BaseUrl string `json:"baseUrl,omitempty"`

	// BatchPath: The path for REST batch requests.
	BatchPath string `json:"batchPath,omitempty"`

	// CanonicalName: Indicates how the API name should be capitalized and
	// split into various parts. Useful for generating pretty class names.
	CanonicalName string `json:"canonicalName,omitempty"`

	// Description: The description of this API.
	Description string `json:"description,omitempty"`

	// DiscoveryVersion: Indicate the version of the Discovery API used to
	// generate this doc.
	DiscoveryVersion string `json:"discoveryVersion,omitempty"`

	// DocumentationLink: A link to human readable documentation for the
	// API.
	DocumentationLink string `json:"documentationLink,omitempty"`

	// Etag: The ETag for this response.
	Etag string `json:"etag,omitempty"`

	// Features: A list of supported features for this API.
	Features []string `json:"features,omitempty"`

	// Icons: Links to 16x16 and 32x32 icons representing the API.
	Icons *RestDescriptionIcons `json:"icons,omitempty"`

	// Id: The ID of this API.
	Id string `json:"id,omitempty"`

	// Kind: The kind for this response.
	Kind string `json:"kind,omitempty"`

	// Labels: Labels for the status of this API, such as labs or
	// deprecated.
	Labels []string `json:"labels,omitempty"`

	// Methods: API-level methods for this API.
	Methods *RestDescriptionMethods `json:"methods,omitempty"`

	// Name: The name of this API.
	Name string `json:"name,omitempty"`

	// OwnerDomain: The domain of the owner of this API. Together with the
	// ownerName and a packagePath values, this can be used to generate a
	// library for this API which would have a unique fully qualified name.
	OwnerDomain string `json:"ownerDomain,omitempty"`

	// OwnerName: The name of the owner of this API. See ownerDomain.
	OwnerName string `json:"ownerName,omitempty"`

	// PackagePath: The package of the owner of this API. See ownerDomain.
	PackagePath string `json:"packagePath,omitempty"`

	// Parameters: Common parameters that apply across all apis.
	Parameters *RestDescriptionParameters `json:"parameters,omitempty"`

	// Protocol: The protocol described by this document.
	Protocol string `json:"protocol,omitempty"`

	// Resources: The resources in this API.
	Resources *RestDescriptionResources `json:"resources,omitempty"`

	// Revision: The version of this API.
	Revision string `json:"revision,omitempty"`

	// RootUrl: The root URL under which all API services live.
	RootUrl string `json:"rootUrl,omitempty"`

	// Schemas: The schemas for this API.
	Schemas *RestDescriptionSchemas `json:"schemas,omitempty"`

	// ServicePath: The base path for all REST requests.
	ServicePath string `json:"servicePath,omitempty"`

	// Title: The title of this API.
	Title string `json:"title,omitempty"`

	// Version: The version of this API.
	Version string `json:"version,omitempty"`
}

type RestDescriptionAuth struct {
	// Oauth2: OAuth 2.0 authentication information.
	Oauth2 *RestDescriptionAuthOauth2 `json:"oauth2,omitempty"`
}

type RestDescriptionAuthOauth2 struct {
	// Scopes: Available OAuth 2.0 scopes.
	Scopes *RestDescriptionAuthOauth2Scopes `json:"scopes,omitempty"`
}

type RestDescriptionAuthOauth2Scopes struct {
}

type RestDescriptionIcons struct {
	// X16: The URL of the 16x16 icon.
	X16 string `json:"x16,omitempty"`

	// X32: The URL of the 32x32 icon.
	X32 string `json:"x32,omitempty"`
}

type RestDescriptionMethods struct {
}

type RestDescriptionParameters struct {
}

type RestDescriptionResources struct {
}

type RestDescriptionSchemas struct {
}

type RestMethod struct {
	// Description: Description of this method.
	Description string `json:"description,omitempty"`

	// EtagRequired: Whether this method requires an ETag to be specified.
	// The ETag is sent as an HTTP If-Match or If-None-Match header.
	EtagRequired bool `json:"etagRequired,omitempty"`

	// HttpMethod: HTTP method used by this method.
	HttpMethod string `json:"httpMethod,omitempty"`

	// Id: A unique ID for this method. This property can be used to match
	// methods between different versions of Discovery.
	Id string `json:"id,omitempty"`

	// MediaUpload: Media upload parameters.
	MediaUpload *RestMethodMediaUpload `json:"mediaUpload,omitempty"`

	// ParameterOrder: Ordered list of required parameters, serves as a hint
	// to clients on how to structure their method signatures. The array is
	// ordered such that the "most-significant" parameter appears first.
	ParameterOrder []string `json:"parameterOrder,omitempty"`

	// Parameters: Details for all parameters in this method.
	Parameters *RestMethodParameters `json:"parameters,omitempty"`

	// Path: The URI path of this REST method. Should be used in conjunction
	// with the basePath property at the api-level.
	Path string `json:"path,omitempty"`

	// Request: The schema for the request.
	Request *RestMethodRequest `json:"request,omitempty"`

	// Response: The schema for the response.
	Response *RestMethodResponse `json:"response,omitempty"`

	// Scopes: OAuth 2.0 scopes applicable to this method.
	Scopes []string `json:"scopes,omitempty"`

	// SupportsMediaDownload: Whether this method supports media downloads.
	SupportsMediaDownload bool `json:"supportsMediaDownload,omitempty"`

	// SupportsMediaUpload: Whether this method supports media uploads.
	SupportsMediaUpload bool `json:"supportsMediaUpload,omitempty"`

	// SupportsSubscription: Whether this method supports subscriptions.
	SupportsSubscription bool `json:"supportsSubscription,omitempty"`
}

type RestMethodMediaUpload struct {
	// Accept: MIME Media Ranges for acceptable media uploads to this
	// method.
	Accept []string `json:"accept,omitempty"`

	// MaxSize: Maximum size of a media upload, such as "1MB", "2GB" or
	// "3TB".
	MaxSize string `json:"maxSize,omitempty"`

	// Protocols: Supported upload protocols.
	Protocols *RestMethodMediaUploadProtocols `json:"protocols,omitempty"`
}

type RestMethodMediaUploadProtocols struct {
	// Resumable: Supports the Resumable Media Upload protocol.
	Resumable *RestMethodMediaUploadProtocolsResumable `json:"resumable,omitempty"`

	// Simple: Supports uploading as a single HTTP request.
	Simple *RestMethodMediaUploadProtocolsSimple `json:"simple,omitempty"`
}

type RestMethodMediaUploadProtocolsResumable struct {
	// Multipart: True if this endpoint supports uploading multipart media.
	Multipart bool `json:"multipart,omitempty"`

	// Path: The URI path to be used for upload. Should be used in
	// conjunction with the basePath property at the api-level.
	Path string `json:"path,omitempty"`
}

type RestMethodMediaUploadProtocolsSimple struct {
	// Multipart: True if this endpoint supports upload multipart media.
	Multipart bool `json:"multipart,omitempty"`

	// Path: The URI path to be used for upload. Should be used in
	// conjunction with the basePath property at the api-level.
	Path string `json:"path,omitempty"`
}

type RestMethodParameters struct {
}

type RestMethodRequest struct {
	// Ref: Schema ID for the request schema.
	Ref string `json:"$ref,omitempty"`

	// ParameterName: parameter name.
	ParameterName string `json:"parameterName,omitempty"`
}

type RestMethodResponse struct {
	// Ref: Schema ID for the response schema.
	Ref string `json:"$ref,omitempty"`
}

type RestResource struct {
	// Methods: Methods on this resource.
	Methods *RestResourceMethods `json:"methods,omitempty"`

	// Resources: Sub-resources on this resource.
	Resources *RestResourceResources `json:"resources,omitempty"`
}

type RestResourceMethods struct {
}

type RestResourceResources struct {
}

// method id "discovery.apis.getRest":

type ApisGetRestCall struct {
	s       *Service
	api     string
	version string
	opt_    map[string]interface{}
}

// GetRest: Retrieve the description of a particular version of an api.
func (r *ApisService) GetRest(api string, version string) *ApisGetRestCall {
	c := &ApisGetRestCall{s: r.s, opt_: make(map[string]interface{})}
	c.api = api
	c.version = version
	return c
}

func (c *ApisGetRestCall) Do() (*RestDescription, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "apis/{api}/{version}/rest")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{api}", url.QueryEscape(c.api), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{version}", url.QueryEscape(c.version), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(RestDescription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieve the description of a particular version of an api.",
	//   "httpMethod": "GET",
	//   "id": "discovery.apis.getRest",
	//   "parameterOrder": [
	//     "api",
	//     "version"
	//   ],
	//   "parameters": {
	//     "api": {
	//       "description": "The name of the API.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "version": {
	//       "description": "The version of the API.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "apis/{api}/{version}/rest",
	//   "response": {
	//     "$ref": "RestDescription"
	//   }
	// }

}

// method id "discovery.apis.list":

type ApisListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieve the list of APIs supported at this endpoint.
func (r *ApisService) List() *ApisListCall {
	c := &ApisListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Name sets the optional parameter "name": Only include APIs with the
// given name.
func (c *ApisListCall) Name(name string) *ApisListCall {
	c.opt_["name"] = name
	return c
}

// Preferred sets the optional parameter "preferred": Return only the
// preferred version of an API.
func (c *ApisListCall) Preferred(preferred bool) *ApisListCall {
	c.opt_["preferred"] = preferred
	return c
}

func (c *ApisListCall) Do() (*DirectoryList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["name"]; ok {
		params.Set("name", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["preferred"]; ok {
		params.Set("preferred", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "apis")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(DirectoryList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieve the list of APIs supported at this endpoint.",
	//   "httpMethod": "GET",
	//   "id": "discovery.apis.list",
	//   "parameters": {
	//     "name": {
	//       "description": "Only include APIs with the given name.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "preferred": {
	//       "default": "false",
	//       "description": "Return only the preferred version of an API.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "apis",
	//   "response": {
	//     "$ref": "DirectoryList"
	//   }
	// }

}
