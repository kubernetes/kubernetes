// Package oauth2 provides access to the Google OAuth2 API.
//
// See https://developers.google.com/accounts/docs/OAuth2
//
// Usage example:
//
//   import "google.golang.org/api/oauth2/v1"
//   ...
//   oauth2Service, err := oauth2.New(oauthHttpClient)
package oauth2 // import "google.golang.org/api/oauth2/v1"

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

const apiId = "oauth2:v1"
const apiName = "oauth2"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// Know the list of people in your circles, your age range, and language
	PlusLoginScope = "https://www.googleapis.com/auth/plus.login"

	// Know who you are on Google
	PlusMeScope = "https://www.googleapis.com/auth/plus.me"

	// View your email address
	UserinfoEmailScope = "https://www.googleapis.com/auth/userinfo.email"

	// View your basic profile info
	UserinfoProfileScope = "https://www.googleapis.com/auth/userinfo.profile"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Userinfo = NewUserinfoService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Userinfo *UserinfoService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewUserinfoService(s *Service) *UserinfoService {
	rs := &UserinfoService{s: s}
	rs.V2 = NewUserinfoV2Service(s)
	return rs
}

type UserinfoService struct {
	s *Service

	V2 *UserinfoV2Service
}

func NewUserinfoV2Service(s *Service) *UserinfoV2Service {
	rs := &UserinfoV2Service{s: s}
	rs.Me = NewUserinfoV2MeService(s)
	return rs
}

type UserinfoV2Service struct {
	s *Service

	Me *UserinfoV2MeService
}

func NewUserinfoV2MeService(s *Service) *UserinfoV2MeService {
	rs := &UserinfoV2MeService{s: s}
	return rs
}

type UserinfoV2MeService struct {
	s *Service
}

type Jwk struct {
	Keys []*JwkKeys `json:"keys,omitempty"`

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

func (s *Jwk) MarshalJSON() ([]byte, error) {
	type noMethod Jwk
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type JwkKeys struct {
	Alg string `json:"alg,omitempty"`

	E string `json:"e,omitempty"`

	Kid string `json:"kid,omitempty"`

	Kty string `json:"kty,omitempty"`

	N string `json:"n,omitempty"`

	Use string `json:"use,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Alg") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Alg") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JwkKeys) MarshalJSON() ([]byte, error) {
	type noMethod JwkKeys
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Raw struct {
	Keyvalues []*RawKeyvalues `json:"keyvalues,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Keyvalues") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Keyvalues") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Raw) MarshalJSON() ([]byte, error) {
	type noMethod Raw
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type RawKeyvalues struct {
	Algorithm string `json:"algorithm,omitempty"`

	Exponent string `json:"exponent,omitempty"`

	Keyid string `json:"keyid,omitempty"`

	Modulus string `json:"modulus,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Algorithm") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Algorithm") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RawKeyvalues) MarshalJSON() ([]byte, error) {
	type noMethod RawKeyvalues
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Tokeninfo struct {
	// AccessType: The access type granted with this token. It can be
	// offline or online.
	AccessType string `json:"access_type,omitempty"`

	// Audience: Who is the intended audience for this token. In general the
	// same as issued_to.
	Audience string `json:"audience,omitempty"`

	// Email: The email address of the user. Present only if the email scope
	// is present in the request.
	Email string `json:"email,omitempty"`

	// EmailVerified: Boolean flag which is true if the email address is
	// verified. Present only if the email scope is present in the request.
	EmailVerified bool `json:"email_verified,omitempty"`

	// ExpiresIn: The expiry time of the token, as number of seconds left
	// until expiry.
	ExpiresIn int64 `json:"expires_in,omitempty"`

	// IssuedAt: The issue time of the token, as number of seconds.
	IssuedAt int64 `json:"issued_at,omitempty"`

	// IssuedTo: To whom was the token issued to. In general the same as
	// audience.
	IssuedTo string `json:"issued_to,omitempty"`

	// Issuer: Who issued the token.
	Issuer string `json:"issuer,omitempty"`

	// Nonce: Nonce of the id token.
	Nonce string `json:"nonce,omitempty"`

	// Scope: The space separated list of scopes granted to this token.
	Scope string `json:"scope,omitempty"`

	// UserId: The obfuscated user id.
	UserId string `json:"user_id,omitempty"`

	// VerifiedEmail: Boolean flag which is true if the email address is
	// verified. Present only if the email scope is present in the request.
	VerifiedEmail bool `json:"verified_email,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccessType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccessType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Tokeninfo) MarshalJSON() ([]byte, error) {
	type noMethod Tokeninfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Userinfoplus struct {
	// Email: The user's email address.
	Email string `json:"email,omitempty"`

	// FamilyName: The user's last name.
	FamilyName string `json:"family_name,omitempty"`

	// Gender: The user's gender.
	Gender string `json:"gender,omitempty"`

	// GivenName: The user's first name.
	GivenName string `json:"given_name,omitempty"`

	// Hd: The hosted domain e.g. example.com if the user is Google apps
	// user.
	Hd string `json:"hd,omitempty"`

	// Id: The obfuscated ID of the user.
	Id string `json:"id,omitempty"`

	// Link: URL of the profile page.
	Link string `json:"link,omitempty"`

	// Locale: The user's preferred locale.
	Locale string `json:"locale,omitempty"`

	// Name: The user's full name.
	Name string `json:"name,omitempty"`

	// Picture: URL of the user's picture image.
	Picture string `json:"picture,omitempty"`

	// VerifiedEmail: Boolean flag which is true if the email address is
	// verified. Always verified because we only return the user's primary
	// email address.
	//
	// Default: true
	VerifiedEmail *bool `json:"verified_email,omitempty"`

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

func (s *Userinfoplus) MarshalJSON() ([]byte, error) {
	type noMethod Userinfoplus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "oauth2.getCertForOpenIdConnect":

type GetCertForOpenIdConnectCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetCertForOpenIdConnect:
func (s *Service) GetCertForOpenIdConnect() *GetCertForOpenIdConnectCall {
	c := &GetCertForOpenIdConnectCall{s: s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GetCertForOpenIdConnectCall) Fields(s ...googleapi.Field) *GetCertForOpenIdConnectCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GetCertForOpenIdConnectCall) IfNoneMatch(entityTag string) *GetCertForOpenIdConnectCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GetCertForOpenIdConnectCall) Context(ctx context.Context) *GetCertForOpenIdConnectCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GetCertForOpenIdConnectCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GetCertForOpenIdConnectCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "oauth2/v1/certs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.getCertForOpenIdConnect" call.
func (c *GetCertForOpenIdConnectCall) Do(opts ...googleapi.CallOption) (map[string]string, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret map[string]string
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "oauth2.getCertForOpenIdConnect",
	//   "path": "oauth2/v1/certs",
	//   "response": {
	//     "$ref": "X509"
	//   }
	// }

}

// method id "oauth2.getCertForOpenIdConnectRaw":

type GetCertForOpenIdConnectRawCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetCertForOpenIdConnectRaw:
func (s *Service) GetCertForOpenIdConnectRaw() *GetCertForOpenIdConnectRawCall {
	c := &GetCertForOpenIdConnectRawCall{s: s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GetCertForOpenIdConnectRawCall) Fields(s ...googleapi.Field) *GetCertForOpenIdConnectRawCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GetCertForOpenIdConnectRawCall) IfNoneMatch(entityTag string) *GetCertForOpenIdConnectRawCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GetCertForOpenIdConnectRawCall) Context(ctx context.Context) *GetCertForOpenIdConnectRawCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GetCertForOpenIdConnectRawCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GetCertForOpenIdConnectRawCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "oauth2/v1/raw_public_keys")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.getCertForOpenIdConnectRaw" call.
// Exactly one of *Raw or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Raw.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *GetCertForOpenIdConnectRawCall) Do(opts ...googleapi.CallOption) (*Raw, error) {
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
	ret := &Raw{
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
	//   "httpMethod": "GET",
	//   "id": "oauth2.getCertForOpenIdConnectRaw",
	//   "path": "oauth2/v1/raw_public_keys",
	//   "response": {
	//     "$ref": "Raw"
	//   }
	// }

}

// method id "oauth2.getRobotJwk":

type GetRobotJwkCall struct {
	s            *Service
	robotEmail   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetRobotJwk:
func (s *Service) GetRobotJwk(robotEmail string) *GetRobotJwkCall {
	c := &GetRobotJwkCall{s: s, urlParams_: make(gensupport.URLParams)}
	c.robotEmail = robotEmail
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GetRobotJwkCall) Fields(s ...googleapi.Field) *GetRobotJwkCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GetRobotJwkCall) IfNoneMatch(entityTag string) *GetRobotJwkCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GetRobotJwkCall) Context(ctx context.Context) *GetRobotJwkCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GetRobotJwkCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GetRobotJwkCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "service_accounts/v1/jwk/{robotEmail}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"robotEmail": c.robotEmail,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.getRobotJwk" call.
// Exactly one of *Jwk or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Jwk.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *GetRobotJwkCall) Do(opts ...googleapi.CallOption) (*Jwk, error) {
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
	ret := &Jwk{
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
	//   "httpMethod": "GET",
	//   "id": "oauth2.getRobotJwk",
	//   "parameterOrder": [
	//     "robotEmail"
	//   ],
	//   "parameters": {
	//     "robotEmail": {
	//       "description": "The email of robot account.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "service_accounts/v1/jwk/{robotEmail}",
	//   "response": {
	//     "$ref": "Jwk"
	//   }
	// }

}

// method id "oauth2.getRobotMetadataRaw":

type GetRobotMetadataRawCall struct {
	s            *Service
	robotEmail   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetRobotMetadataRaw:
func (s *Service) GetRobotMetadataRaw(robotEmail string) *GetRobotMetadataRawCall {
	c := &GetRobotMetadataRawCall{s: s, urlParams_: make(gensupport.URLParams)}
	c.robotEmail = robotEmail
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GetRobotMetadataRawCall) Fields(s ...googleapi.Field) *GetRobotMetadataRawCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GetRobotMetadataRawCall) IfNoneMatch(entityTag string) *GetRobotMetadataRawCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GetRobotMetadataRawCall) Context(ctx context.Context) *GetRobotMetadataRawCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GetRobotMetadataRawCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GetRobotMetadataRawCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "service_accounts/v1/metadata/raw/{robotEmail}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"robotEmail": c.robotEmail,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.getRobotMetadataRaw" call.
// Exactly one of *Raw or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Raw.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *GetRobotMetadataRawCall) Do(opts ...googleapi.CallOption) (*Raw, error) {
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
	ret := &Raw{
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
	//   "httpMethod": "GET",
	//   "id": "oauth2.getRobotMetadataRaw",
	//   "parameterOrder": [
	//     "robotEmail"
	//   ],
	//   "parameters": {
	//     "robotEmail": {
	//       "description": "The email of robot account.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "service_accounts/v1/metadata/raw/{robotEmail}",
	//   "response": {
	//     "$ref": "Raw"
	//   }
	// }

}

// method id "oauth2.getRobotMetadataX509":

type GetRobotMetadataX509Call struct {
	s            *Service
	robotEmail   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetRobotMetadataX509:
func (s *Service) GetRobotMetadataX509(robotEmail string) *GetRobotMetadataX509Call {
	c := &GetRobotMetadataX509Call{s: s, urlParams_: make(gensupport.URLParams)}
	c.robotEmail = robotEmail
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GetRobotMetadataX509Call) Fields(s ...googleapi.Field) *GetRobotMetadataX509Call {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GetRobotMetadataX509Call) IfNoneMatch(entityTag string) *GetRobotMetadataX509Call {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GetRobotMetadataX509Call) Context(ctx context.Context) *GetRobotMetadataX509Call {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GetRobotMetadataX509Call) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GetRobotMetadataX509Call) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "service_accounts/v1/metadata/x509/{robotEmail}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"robotEmail": c.robotEmail,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.getRobotMetadataX509" call.
func (c *GetRobotMetadataX509Call) Do(opts ...googleapi.CallOption) (map[string]string, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret map[string]string
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "oauth2.getRobotMetadataX509",
	//   "parameterOrder": [
	//     "robotEmail"
	//   ],
	//   "parameters": {
	//     "robotEmail": {
	//       "description": "The email of robot account.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "service_accounts/v1/metadata/x509/{robotEmail}",
	//   "response": {
	//     "$ref": "X509"
	//   }
	// }

}

// method id "oauth2.tokeninfo":

type TokeninfoCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Tokeninfo: Get token info
func (s *Service) Tokeninfo() *TokeninfoCall {
	c := &TokeninfoCall{s: s, urlParams_: make(gensupport.URLParams)}
	return c
}

// AccessToken sets the optional parameter "access_token": The oauth2
// access token
func (c *TokeninfoCall) AccessToken(accessToken string) *TokeninfoCall {
	c.urlParams_.Set("access_token", accessToken)
	return c
}

// IdToken sets the optional parameter "id_token": The ID token
func (c *TokeninfoCall) IdToken(idToken string) *TokeninfoCall {
	c.urlParams_.Set("id_token", idToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TokeninfoCall) Fields(s ...googleapi.Field) *TokeninfoCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TokeninfoCall) Context(ctx context.Context) *TokeninfoCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TokeninfoCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TokeninfoCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "oauth2/v1/tokeninfo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.tokeninfo" call.
// Exactly one of *Tokeninfo or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Tokeninfo.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TokeninfoCall) Do(opts ...googleapi.CallOption) (*Tokeninfo, error) {
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
	ret := &Tokeninfo{
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
	//   "description": "Get token info",
	//   "httpMethod": "POST",
	//   "id": "oauth2.tokeninfo",
	//   "parameters": {
	//     "access_token": {
	//       "description": "The oauth2 access token",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id_token": {
	//       "description": "The ID token",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "oauth2/v1/tokeninfo",
	//   "response": {
	//     "$ref": "Tokeninfo"
	//   }
	// }

}

// method id "oauth2.userinfo.get":

type UserinfoGetCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get user info
func (r *UserinfoService) Get() *UserinfoGetCall {
	c := &UserinfoGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserinfoGetCall) Fields(s ...googleapi.Field) *UserinfoGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UserinfoGetCall) IfNoneMatch(entityTag string) *UserinfoGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserinfoGetCall) Context(ctx context.Context) *UserinfoGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UserinfoGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UserinfoGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "oauth2/v1/userinfo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.userinfo.get" call.
// Exactly one of *Userinfoplus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Userinfoplus.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UserinfoGetCall) Do(opts ...googleapi.CallOption) (*Userinfoplus, error) {
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
	ret := &Userinfoplus{
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
	//   "description": "Get user info",
	//   "httpMethod": "GET",
	//   "id": "oauth2.userinfo.get",
	//   "path": "oauth2/v1/userinfo",
	//   "response": {
	//     "$ref": "Userinfoplus"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me",
	//     "https://www.googleapis.com/auth/userinfo.email",
	//     "https://www.googleapis.com/auth/userinfo.profile"
	//   ]
	// }

}

// method id "oauth2.userinfo.v2.me.get":

type UserinfoV2MeGetCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get user info
func (r *UserinfoV2MeService) Get() *UserinfoV2MeGetCall {
	c := &UserinfoV2MeGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserinfoV2MeGetCall) Fields(s ...googleapi.Field) *UserinfoV2MeGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UserinfoV2MeGetCall) IfNoneMatch(entityTag string) *UserinfoV2MeGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserinfoV2MeGetCall) Context(ctx context.Context) *UserinfoV2MeGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UserinfoV2MeGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UserinfoV2MeGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "userinfo/v2/me")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "oauth2.userinfo.v2.me.get" call.
// Exactly one of *Userinfoplus or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Userinfoplus.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UserinfoV2MeGetCall) Do(opts ...googleapi.CallOption) (*Userinfoplus, error) {
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
	ret := &Userinfoplus{
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
	//   "description": "Get user info",
	//   "httpMethod": "GET",
	//   "id": "oauth2.userinfo.v2.me.get",
	//   "path": "userinfo/v2/me",
	//   "response": {
	//     "$ref": "Userinfoplus"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me",
	//     "https://www.googleapis.com/auth/userinfo.email",
	//     "https://www.googleapis.com/auth/userinfo.profile"
	//   ]
	// }

}
