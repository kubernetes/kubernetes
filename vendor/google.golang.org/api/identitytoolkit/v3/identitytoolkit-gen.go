// Package identitytoolkit provides access to the Google Identity Toolkit API.
//
// See https://developers.google.com/identity-toolkit/v3/
//
// Usage example:
//
//   import "google.golang.org/api/identitytoolkit/v3"
//   ...
//   identitytoolkitService, err := identitytoolkit.New(oauthHttpClient)
package identitytoolkit // import "google.golang.org/api/identitytoolkit/v3"

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

const apiId = "identitytoolkit:v3"
const apiName = "identitytoolkit"
const apiVersion = "v3"
const basePath = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Relyingparty = NewRelyingpartyService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Relyingparty *RelyingpartyService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewRelyingpartyService(s *Service) *RelyingpartyService {
	rs := &RelyingpartyService{s: s}
	return rs
}

type RelyingpartyService struct {
	s *Service
}

// CreateAuthUriResponse: Response of creating the IDP authentication
// URL.
type CreateAuthUriResponse struct {
	// AuthUri: The URI used by the IDP to authenticate the user.
	AuthUri string `json:"authUri,omitempty"`

	// CaptchaRequired: True if captcha is required.
	CaptchaRequired bool `json:"captchaRequired,omitempty"`

	// ForExistingProvider: True if the authUri is for user's existing
	// provider.
	ForExistingProvider bool `json:"forExistingProvider,omitempty"`

	// Kind: The fixed string identitytoolkit#CreateAuthUriResponse".
	Kind string `json:"kind,omitempty"`

	// ProviderId: The provider ID of the auth URI.
	ProviderId string `json:"providerId,omitempty"`

	// Registered: Whether the user is registered if the identifier is an
	// email.
	Registered bool `json:"registered,omitempty"`

	// SessionId: Session ID which should be passed in the following
	// verifyAssertion request.
	SessionId string `json:"sessionId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AuthUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreateAuthUriResponse) MarshalJSON() ([]byte, error) {
	type noMethod CreateAuthUriResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DeleteAccountResponse: Respone of deleting account.
type DeleteAccountResponse struct {
	// Kind: The fixed string "identitytoolkit#DeleteAccountResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeleteAccountResponse) MarshalJSON() ([]byte, error) {
	type noMethod DeleteAccountResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DownloadAccountResponse: Respone of downloading accounts in batch.
type DownloadAccountResponse struct {
	// Kind: The fixed string "identitytoolkit#DownloadAccountResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The next page token. To be used in a subsequent
	// request to return the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Users: The user accounts data.
	Users []*UserInfo `json:"users,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DownloadAccountResponse) MarshalJSON() ([]byte, error) {
	type noMethod DownloadAccountResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GetAccountInfoResponse: Response of getting account information.
type GetAccountInfoResponse struct {
	// Kind: The fixed string "identitytoolkit#GetAccountInfoResponse".
	Kind string `json:"kind,omitempty"`

	// Users: The info of the users.
	Users []*UserInfo `json:"users,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetAccountInfoResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetAccountInfoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GetOobConfirmationCodeResponse: Response of getting a code for user
// confirmation (reset password, change email etc.).
type GetOobConfirmationCodeResponse struct {
	// Email: The email address that the email is sent to.
	Email string `json:"email,omitempty"`

	// Kind: The fixed string
	// "identitytoolkit#GetOobConfirmationCodeResponse".
	Kind string `json:"kind,omitempty"`

	// OobCode: The code to be send to the user.
	OobCode string `json:"oobCode,omitempty"`

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
}

func (s *GetOobConfirmationCodeResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetOobConfirmationCodeResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GetRecaptchaParamResponse: Response of getting recaptcha param.
type GetRecaptchaParamResponse struct {
	// Kind: The fixed string "identitytoolkit#GetRecaptchaParamResponse".
	Kind string `json:"kind,omitempty"`

	// RecaptchaSiteKey: Site key registered at recaptcha.
	RecaptchaSiteKey string `json:"recaptchaSiteKey,omitempty"`

	// RecaptchaStoken: The stoken field for the recaptcha widget, used to
	// request captcha challenge.
	RecaptchaStoken string `json:"recaptchaStoken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetRecaptchaParamResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetRecaptchaParamResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyCreateAuthUriRequest: Request to get the
// IDP authentication URL.
type IdentitytoolkitRelyingpartyCreateAuthUriRequest struct {
	// AppId: The app ID of the mobile app, base64(CERT_SHA1):PACKAGE_NAME
	// for Android, BUNDLE_ID for iOS.
	AppId string `json:"appId,omitempty"`

	// ClientId: The relying party OAuth client ID.
	ClientId string `json:"clientId,omitempty"`

	// Context: The opaque value used by the client to maintain context info
	// between the authentication request and the IDP callback.
	Context string `json:"context,omitempty"`

	// ContinueUri: The URI to which the IDP redirects the user after the
	// federated login flow.
	ContinueUri string `json:"continueUri,omitempty"`

	// Identifier: The email or federated ID of the user.
	Identifier string `json:"identifier,omitempty"`

	// OauthConsumerKey: The developer's consumer key for OpenId OAuth
	// Extension
	OauthConsumerKey string `json:"oauthConsumerKey,omitempty"`

	// OauthScope: Additional oauth scopes, beyond the basid user profile,
	// that the user would be prompted to grant
	OauthScope string `json:"oauthScope,omitempty"`

	// OpenidRealm: Optional realm for OpenID protocol. The sub string
	// "scheme://domain:port" of the param "continueUri" is used if this is
	// not set.
	OpenidRealm string `json:"openidRealm,omitempty"`

	// OtaApp: The native app package for OTA installation.
	OtaApp string `json:"otaApp,omitempty"`

	// ProviderId: The IdP ID. For white listed IdPs it's a short domain
	// name e.g. google.com, aol.com, live.net and yahoo.com. For other
	// OpenID IdPs it's the OP identifier.
	ProviderId string `json:"providerId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AppId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyCreateAuthUriRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyCreateAuthUriRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyDeleteAccountRequest: Request to delete
// account.
type IdentitytoolkitRelyingpartyDeleteAccountRequest struct {
	// LocalId: The local ID of the user.
	LocalId string `json:"localId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LocalId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyDeleteAccountRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyDeleteAccountRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyDownloadAccountRequest: Request to
// download user account in batch.
type IdentitytoolkitRelyingpartyDownloadAccountRequest struct {
	// MaxResults: The max number of results to return in the response.
	MaxResults int64 `json:"maxResults,omitempty"`

	// NextPageToken: The token for the next page. This should be taken from
	// the previous response.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxResults") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyDownloadAccountRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyDownloadAccountRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyGetAccountInfoRequest: Request to get the
// account information.
type IdentitytoolkitRelyingpartyGetAccountInfoRequest struct {
	// Email: The list of emails of the users to inquiry.
	Email []string `json:"email,omitempty"`

	// IdToken: The GITKit token of the authenticated user.
	IdToken string `json:"idToken,omitempty"`

	// LocalId: The list of local ID's of the users to inquiry.
	LocalId []string `json:"localId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Email") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyGetAccountInfoRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyGetAccountInfoRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyGetProjectConfigResponse: Response of
// getting the project configuration.
type IdentitytoolkitRelyingpartyGetProjectConfigResponse struct {
	// AllowPasswordUser: Whether to allow password user sign in or sign up.
	AllowPasswordUser bool `json:"allowPasswordUser,omitempty"`

	// ApiKey: Browser API key, needed when making http request to Apiary.
	ApiKey string `json:"apiKey,omitempty"`

	// IdpConfig: OAuth2 provider configuration.
	IdpConfig []*IdpConfig `json:"idpConfig,omitempty"`

	// ProjectId: Project ID of the relying party.
	ProjectId string `json:"projectId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AllowPasswordUser")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyGetProjectConfigResponse) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyGetProjectConfigResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyResetPasswordRequest: Request to reset the
// password.
type IdentitytoolkitRelyingpartyResetPasswordRequest struct {
	// Email: The email address of the user.
	Email string `json:"email,omitempty"`

	// NewPassword: The new password inputted by the user.
	NewPassword string `json:"newPassword,omitempty"`

	// OldPassword: The old password inputted by the user.
	OldPassword string `json:"oldPassword,omitempty"`

	// OobCode: The confirmation code.
	OobCode string `json:"oobCode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Email") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyResetPasswordRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyResetPasswordRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartySetAccountInfoRequest: Request to set the
// account information.
type IdentitytoolkitRelyingpartySetAccountInfoRequest struct {
	// CaptchaChallenge: The captcha challenge.
	CaptchaChallenge string `json:"captchaChallenge,omitempty"`

	// CaptchaResponse: Response to the captcha.
	CaptchaResponse string `json:"captchaResponse,omitempty"`

	// DisableUser: Whether to disable the user.
	DisableUser bool `json:"disableUser,omitempty"`

	// DisplayName: The name of the user.
	DisplayName string `json:"displayName,omitempty"`

	// Email: The email of the user.
	Email string `json:"email,omitempty"`

	// EmailVerified: Mark the email as verified or not.
	EmailVerified bool `json:"emailVerified,omitempty"`

	// IdToken: The GITKit token of the authenticated user.
	IdToken string `json:"idToken,omitempty"`

	// LocalId: The local ID of the user.
	LocalId string `json:"localId,omitempty"`

	// OobCode: The out-of-band code of the change email request.
	OobCode string `json:"oobCode,omitempty"`

	// Password: The new password of the user.
	Password string `json:"password,omitempty"`

	// Provider: The associated IDPs of the user.
	Provider []string `json:"provider,omitempty"`

	// UpgradeToFederatedLogin: Mark the user to upgrade to federated login.
	UpgradeToFederatedLogin bool `json:"upgradeToFederatedLogin,omitempty"`

	// ValidSince: Timestamp in seconds for valid login token.
	ValidSince int64 `json:"validSince,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "CaptchaChallenge") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartySetAccountInfoRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartySetAccountInfoRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyUploadAccountRequest: Request to upload
// user account in batch.
type IdentitytoolkitRelyingpartyUploadAccountRequest struct {
	// HashAlgorithm: The password hash algorithm.
	HashAlgorithm string `json:"hashAlgorithm,omitempty"`

	// MemoryCost: Memory cost for hash calculation. Used by scrypt similar
	// algorithms.
	MemoryCost int64 `json:"memoryCost,omitempty"`

	// Rounds: Rounds for hash calculation. Used by scrypt and similar
	// algorithms.
	Rounds int64 `json:"rounds,omitempty"`

	// SaltSeparator: The salt separator.
	SaltSeparator string `json:"saltSeparator,omitempty"`

	// SignerKey: The key for to hash the password.
	SignerKey string `json:"signerKey,omitempty"`

	// Users: The account info to be stored.
	Users []*UserInfo `json:"users,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HashAlgorithm") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyUploadAccountRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyUploadAccountRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyVerifyAssertionRequest: Request to verify
// the IDP assertion.
type IdentitytoolkitRelyingpartyVerifyAssertionRequest struct {
	// PendingIdToken: The GITKit token for the non-trusted IDP pending to
	// be confirmed by the user.
	PendingIdToken string `json:"pendingIdToken,omitempty"`

	// PostBody: The post body if the request is a HTTP POST.
	PostBody string `json:"postBody,omitempty"`

	// RequestUri: The URI to which the IDP redirects the user back. It may
	// contain federated login result params added by the IDP.
	RequestUri string `json:"requestUri,omitempty"`

	// ReturnRefreshToken: Whether to return refresh tokens.
	ReturnRefreshToken bool `json:"returnRefreshToken,omitempty"`

	// SessionId: Session ID, which should match the one in previous
	// createAuthUri request.
	SessionId string `json:"sessionId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PendingIdToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyVerifyAssertionRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyVerifyAssertionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdentitytoolkitRelyingpartyVerifyPasswordRequest: Request to verify
// the password.
type IdentitytoolkitRelyingpartyVerifyPasswordRequest struct {
	// CaptchaChallenge: The captcha challenge.
	CaptchaChallenge string `json:"captchaChallenge,omitempty"`

	// CaptchaResponse: Response to the captcha.
	CaptchaResponse string `json:"captchaResponse,omitempty"`

	// Email: The email of the user.
	Email string `json:"email,omitempty"`

	// Password: The password inputed by the user.
	Password string `json:"password,omitempty"`

	// PendingIdToken: The GITKit token for the non-trusted IDP, which is to
	// be confirmed by the user.
	PendingIdToken string `json:"pendingIdToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CaptchaChallenge") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdentitytoolkitRelyingpartyVerifyPasswordRequest) MarshalJSON() ([]byte, error) {
	type noMethod IdentitytoolkitRelyingpartyVerifyPasswordRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IdpConfig: Template for a single idp configuration.
type IdpConfig struct {
	// ClientId: OAuth2 client ID.
	ClientId string `json:"clientId,omitempty"`

	// Enabled: Whether this IDP is enabled.
	Enabled bool `json:"enabled,omitempty"`

	// ExperimentPercent: Percent of users who will be prompted/redirected
	// federated login for this IDP.
	ExperimentPercent int64 `json:"experimentPercent,omitempty"`

	// Provider: OAuth2 provider.
	Provider string `json:"provider,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClientId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IdpConfig) MarshalJSON() ([]byte, error) {
	type noMethod IdpConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Relyingparty: Request of getting a code for user confirmation (reset
// password, change email etc.)
type Relyingparty struct {
	// CaptchaResp: The recaptcha response from the user.
	CaptchaResp string `json:"captchaResp,omitempty"`

	// Challenge: The recaptcha challenge presented to the user.
	Challenge string `json:"challenge,omitempty"`

	// Email: The email of the user.
	Email string `json:"email,omitempty"`

	// IdToken: The user's Gitkit login token for email change.
	IdToken string `json:"idToken,omitempty"`

	// Kind: The fixed string "identitytoolkit#relyingparty".
	Kind string `json:"kind,omitempty"`

	// NewEmail: The new email if the code is for email change.
	NewEmail string `json:"newEmail,omitempty"`

	// RequestType: The request type.
	RequestType string `json:"requestType,omitempty"`

	// UserIp: The IP address of the user.
	UserIp string `json:"userIp,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CaptchaResp") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Relyingparty) MarshalJSON() ([]byte, error) {
	type noMethod Relyingparty
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ResetPasswordResponse: Response of resetting the password.
type ResetPasswordResponse struct {
	// Email: The user's email.
	Email string `json:"email,omitempty"`

	// Kind: The fixed string "identitytoolkit#ResetPasswordResponse".
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
}

func (s *ResetPasswordResponse) MarshalJSON() ([]byte, error) {
	type noMethod ResetPasswordResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SetAccountInfoResponse: Respone of setting the account information.
type SetAccountInfoResponse struct {
	// DisplayName: The name of the user.
	DisplayName string `json:"displayName,omitempty"`

	// Email: The email of the user.
	Email string `json:"email,omitempty"`

	// IdToken: The Gitkit id token to login the newly sign up user.
	IdToken string `json:"idToken,omitempty"`

	// Kind: The fixed string "identitytoolkit#SetAccountInfoResponse".
	Kind string `json:"kind,omitempty"`

	// NewEmail: The new email the user attempts to change to.
	NewEmail string `json:"newEmail,omitempty"`

	// ProviderUserInfo: The user's profiles at the associated IdPs.
	ProviderUserInfo []*SetAccountInfoResponseProviderUserInfo `json:"providerUserInfo,omitempty"`

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
}

func (s *SetAccountInfoResponse) MarshalJSON() ([]byte, error) {
	type noMethod SetAccountInfoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type SetAccountInfoResponseProviderUserInfo struct {
	// DisplayName: The user's display name at the IDP.
	DisplayName string `json:"displayName,omitempty"`

	// PhotoUrl: The user's photo url at the IDP.
	PhotoUrl string `json:"photoUrl,omitempty"`

	// ProviderId: The IdP ID. For whitelisted IdPs it's a short domain
	// name, e.g., google.com, aol.com, live.net and yahoo.com. For other
	// OpenID IdPs it's the OP identifier.
	ProviderId string `json:"providerId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SetAccountInfoResponseProviderUserInfo) MarshalJSON() ([]byte, error) {
	type noMethod SetAccountInfoResponseProviderUserInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UploadAccountResponse: Respone of uploading accounts in batch.
type UploadAccountResponse struct {
	// Error: The error encountered while processing the account info.
	Error []*UploadAccountResponseError `json:"error,omitempty"`

	// Kind: The fixed string "identitytoolkit#UploadAccountResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Error") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UploadAccountResponse) MarshalJSON() ([]byte, error) {
	type noMethod UploadAccountResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type UploadAccountResponseError struct {
	// Index: The index of the malformed account, starting from 0.
	Index int64 `json:"index,omitempty"`

	// Message: Detailed error message for the account info.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Index") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UploadAccountResponseError) MarshalJSON() ([]byte, error) {
	type noMethod UploadAccountResponseError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UserInfo: Template for an individual account info.
type UserInfo struct {
	// Disabled: Whether the user is disabled.
	Disabled bool `json:"disabled,omitempty"`

	// DisplayName: The name of the user.
	DisplayName string `json:"displayName,omitempty"`

	// Email: The email of the user.
	Email string `json:"email,omitempty"`

	// EmailVerified: Whether the email has been verified.
	EmailVerified bool `json:"emailVerified,omitempty"`

	// LocalId: The local ID of the user.
	LocalId string `json:"localId,omitempty"`

	// PasswordHash: The user's hashed password.
	PasswordHash string `json:"passwordHash,omitempty"`

	// PasswordUpdatedAt: The timestamp when the password was last updated.
	PasswordUpdatedAt float64 `json:"passwordUpdatedAt,omitempty"`

	// PhotoUrl: The URL of the user profile photo.
	PhotoUrl string `json:"photoUrl,omitempty"`

	// ProviderUserInfo: The IDP of the user.
	ProviderUserInfo []*UserInfoProviderUserInfo `json:"providerUserInfo,omitempty"`

	// Salt: The user's password salt.
	Salt string `json:"salt,omitempty"`

	// ValidSince: Timestamp in seconds for valid login token.
	ValidSince int64 `json:"validSince,omitempty,string"`

	// Version: Version of the user's password.
	Version int64 `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Disabled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UserInfo) MarshalJSON() ([]byte, error) {
	type noMethod UserInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type UserInfoProviderUserInfo struct {
	// DisplayName: The user's display name at the IDP.
	DisplayName string `json:"displayName,omitempty"`

	// FederatedId: User's identifier at IDP.
	FederatedId string `json:"federatedId,omitempty"`

	// PhotoUrl: The user's photo url at the IDP.
	PhotoUrl string `json:"photoUrl,omitempty"`

	// ProviderId: The IdP ID. For white listed IdPs it's a short domain
	// name, e.g., google.com, aol.com, live.net and yahoo.com. For other
	// OpenID IdPs it's the OP identifier.
	ProviderId string `json:"providerId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UserInfoProviderUserInfo) MarshalJSON() ([]byte, error) {
	type noMethod UserInfoProviderUserInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// VerifyAssertionResponse: Response of verifying the IDP assertion.
type VerifyAssertionResponse struct {
	// Action: The action code.
	Action string `json:"action,omitempty"`

	// AppInstallationUrl: URL for OTA app installation.
	AppInstallationUrl string `json:"appInstallationUrl,omitempty"`

	// AppScheme: The custom scheme used by mobile app.
	AppScheme string `json:"appScheme,omitempty"`

	// Context: The opaque value used by the client to maintain context info
	// between the authentication request and the IDP callback.
	Context string `json:"context,omitempty"`

	// DateOfBirth: The birth date of the IdP account.
	DateOfBirth string `json:"dateOfBirth,omitempty"`

	// DisplayName: The display name of the user.
	DisplayName string `json:"displayName,omitempty"`

	// Email: The email returned by the IdP. NOTE: The federated login user
	// may not own the email.
	Email string `json:"email,omitempty"`

	// EmailRecycled: It's true if the email is recycled.
	EmailRecycled bool `json:"emailRecycled,omitempty"`

	// EmailVerified: The value is true if the IDP is also the email
	// provider. It means the user owns the email.
	EmailVerified bool `json:"emailVerified,omitempty"`

	// FederatedId: The unique ID identifies the IdP account.
	FederatedId string `json:"federatedId,omitempty"`

	// FirstName: The first name of the user.
	FirstName string `json:"firstName,omitempty"`

	// FullName: The full name of the user.
	FullName string `json:"fullName,omitempty"`

	// IdToken: The ID token.
	IdToken string `json:"idToken,omitempty"`

	// InputEmail: It's the identifier param in the createAuthUri request if
	// the identifier is an email. It can be used to check whether the user
	// input email is different from the asserted email.
	InputEmail string `json:"inputEmail,omitempty"`

	// Kind: The fixed string "identitytoolkit#VerifyAssertionResponse".
	Kind string `json:"kind,omitempty"`

	// Language: The language preference of the user.
	Language string `json:"language,omitempty"`

	// LastName: The last name of the user.
	LastName string `json:"lastName,omitempty"`

	// LocalId: The RP local ID if it's already been mapped to the IdP
	// account identified by the federated ID.
	LocalId string `json:"localId,omitempty"`

	// NeedConfirmation: Whether the assertion is from a non-trusted IDP and
	// need account linking confirmation.
	NeedConfirmation bool `json:"needConfirmation,omitempty"`

	// NeedEmail: Whether need client to supply email to complete the
	// federated login flow.
	NeedEmail bool `json:"needEmail,omitempty"`

	// NickName: The nick name of the user.
	NickName string `json:"nickName,omitempty"`

	// OauthAccessToken: The OAuth2 access token.
	OauthAccessToken string `json:"oauthAccessToken,omitempty"`

	// OauthAuthorizationCode: The OAuth2 authorization code.
	OauthAuthorizationCode string `json:"oauthAuthorizationCode,omitempty"`

	// OauthExpireIn: The lifetime in seconds of the OAuth2 access token.
	OauthExpireIn int64 `json:"oauthExpireIn,omitempty"`

	// OauthRequestToken: The user approved request token for the OpenID
	// OAuth extension.
	OauthRequestToken string `json:"oauthRequestToken,omitempty"`

	// OauthScope: The scope for the OpenID OAuth extension.
	OauthScope string `json:"oauthScope,omitempty"`

	// OriginalEmail: The original email stored in the mapping storage. It's
	// returned when the federated ID is associated to a different email.
	OriginalEmail string `json:"originalEmail,omitempty"`

	// PhotoUrl: The URI of the public accessible profiel picture.
	PhotoUrl string `json:"photoUrl,omitempty"`

	// ProviderId: The IdP ID. For white listed IdPs it's a short domain
	// name e.g. google.com, aol.com, live.net and yahoo.com. If the
	// "providerId" param is set to OpenID OP identifer other than the
	// whilte listed IdPs the OP identifier is returned. If the "identifier"
	// param is federated ID in the createAuthUri request. The domain part
	// of the federated ID is returned.
	ProviderId string `json:"providerId,omitempty"`

	// TimeZone: The timezone of the user.
	TimeZone string `json:"timeZone,omitempty"`

	// VerifiedProvider: When action is 'map', contains the idps which can
	// be used for confirmation.
	VerifiedProvider []string `json:"verifiedProvider,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Action") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *VerifyAssertionResponse) MarshalJSON() ([]byte, error) {
	type noMethod VerifyAssertionResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// VerifyPasswordResponse: Request of verifying the password.
type VerifyPasswordResponse struct {
	// DisplayName: The name of the user.
	DisplayName string `json:"displayName,omitempty"`

	// Email: The email returned by the IdP. NOTE: The federated login user
	// may not own the email.
	Email string `json:"email,omitempty"`

	// IdToken: The GITKit token for authenticated user.
	IdToken string `json:"idToken,omitempty"`

	// Kind: The fixed string "identitytoolkit#VerifyPasswordResponse".
	Kind string `json:"kind,omitempty"`

	// LocalId: The RP local ID if it's already been mapped to the IdP
	// account identified by the federated ID.
	LocalId string `json:"localId,omitempty"`

	// OauthAccessToken: The OAuth2 access token.
	OauthAccessToken string `json:"oauthAccessToken,omitempty"`

	// OauthAuthorizationCode: The OAuth2 authorization code.
	OauthAuthorizationCode string `json:"oauthAuthorizationCode,omitempty"`

	// OauthExpireIn: The lifetime in seconds of the OAuth2 access token.
	OauthExpireIn int64 `json:"oauthExpireIn,omitempty"`

	// PhotoUrl: The URI of the user's photo at IdP
	PhotoUrl string `json:"photoUrl,omitempty"`

	// Registered: Whether the email is registered.
	Registered bool `json:"registered,omitempty"`

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
}

func (s *VerifyPasswordResponse) MarshalJSON() ([]byte, error) {
	type noMethod VerifyPasswordResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "identitytoolkit.relyingparty.createAuthUri":

type RelyingpartyCreateAuthUriCall struct {
	s                                               *Service
	identitytoolkitrelyingpartycreateauthurirequest *IdentitytoolkitRelyingpartyCreateAuthUriRequest
	urlParams_                                      gensupport.URLParams
	ctx_                                            context.Context
}

// CreateAuthUri: Creates the URI used by the IdP to authenticate the
// user.
func (r *RelyingpartyService) CreateAuthUri(identitytoolkitrelyingpartycreateauthurirequest *IdentitytoolkitRelyingpartyCreateAuthUriRequest) *RelyingpartyCreateAuthUriCall {
	c := &RelyingpartyCreateAuthUriCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartycreateauthurirequest = identitytoolkitrelyingpartycreateauthurirequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyCreateAuthUriCall) QuotaUser(quotaUser string) *RelyingpartyCreateAuthUriCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyCreateAuthUriCall) UserIP(userIP string) *RelyingpartyCreateAuthUriCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyCreateAuthUriCall) Fields(s ...googleapi.Field) *RelyingpartyCreateAuthUriCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyCreateAuthUriCall) Context(ctx context.Context) *RelyingpartyCreateAuthUriCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyCreateAuthUriCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartycreateauthurirequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "createAuthUri")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.createAuthUri" call.
// Exactly one of *CreateAuthUriResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *CreateAuthUriResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyCreateAuthUriCall) Do() (*CreateAuthUriResponse, error) {
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
	ret := &CreateAuthUriResponse{
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
	//   "description": "Creates the URI used by the IdP to authenticate the user.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.createAuthUri",
	//   "path": "createAuthUri",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyCreateAuthUriRequest"
	//   },
	//   "response": {
	//     "$ref": "CreateAuthUriResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.deleteAccount":

type RelyingpartyDeleteAccountCall struct {
	s                                               *Service
	identitytoolkitrelyingpartydeleteaccountrequest *IdentitytoolkitRelyingpartyDeleteAccountRequest
	urlParams_                                      gensupport.URLParams
	ctx_                                            context.Context
}

// DeleteAccount: Delete user account.
func (r *RelyingpartyService) DeleteAccount(identitytoolkitrelyingpartydeleteaccountrequest *IdentitytoolkitRelyingpartyDeleteAccountRequest) *RelyingpartyDeleteAccountCall {
	c := &RelyingpartyDeleteAccountCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartydeleteaccountrequest = identitytoolkitrelyingpartydeleteaccountrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyDeleteAccountCall) QuotaUser(quotaUser string) *RelyingpartyDeleteAccountCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyDeleteAccountCall) UserIP(userIP string) *RelyingpartyDeleteAccountCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyDeleteAccountCall) Fields(s ...googleapi.Field) *RelyingpartyDeleteAccountCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyDeleteAccountCall) Context(ctx context.Context) *RelyingpartyDeleteAccountCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyDeleteAccountCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartydeleteaccountrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "deleteAccount")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.deleteAccount" call.
// Exactly one of *DeleteAccountResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DeleteAccountResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyDeleteAccountCall) Do() (*DeleteAccountResponse, error) {
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
	ret := &DeleteAccountResponse{
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
	//   "description": "Delete user account.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.deleteAccount",
	//   "path": "deleteAccount",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyDeleteAccountRequest"
	//   },
	//   "response": {
	//     "$ref": "DeleteAccountResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.downloadAccount":

type RelyingpartyDownloadAccountCall struct {
	s                                                 *Service
	identitytoolkitrelyingpartydownloadaccountrequest *IdentitytoolkitRelyingpartyDownloadAccountRequest
	urlParams_                                        gensupport.URLParams
	ctx_                                              context.Context
}

// DownloadAccount: Batch download user accounts.
func (r *RelyingpartyService) DownloadAccount(identitytoolkitrelyingpartydownloadaccountrequest *IdentitytoolkitRelyingpartyDownloadAccountRequest) *RelyingpartyDownloadAccountCall {
	c := &RelyingpartyDownloadAccountCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartydownloadaccountrequest = identitytoolkitrelyingpartydownloadaccountrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyDownloadAccountCall) QuotaUser(quotaUser string) *RelyingpartyDownloadAccountCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyDownloadAccountCall) UserIP(userIP string) *RelyingpartyDownloadAccountCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyDownloadAccountCall) Fields(s ...googleapi.Field) *RelyingpartyDownloadAccountCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyDownloadAccountCall) Context(ctx context.Context) *RelyingpartyDownloadAccountCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyDownloadAccountCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartydownloadaccountrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "downloadAccount")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.downloadAccount" call.
// Exactly one of *DownloadAccountResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DownloadAccountResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyDownloadAccountCall) Do() (*DownloadAccountResponse, error) {
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
	ret := &DownloadAccountResponse{
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
	//   "description": "Batch download user accounts.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.downloadAccount",
	//   "path": "downloadAccount",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyDownloadAccountRequest"
	//   },
	//   "response": {
	//     "$ref": "DownloadAccountResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.getAccountInfo":

type RelyingpartyGetAccountInfoCall struct {
	s                                                *Service
	identitytoolkitrelyingpartygetaccountinforequest *IdentitytoolkitRelyingpartyGetAccountInfoRequest
	urlParams_                                       gensupport.URLParams
	ctx_                                             context.Context
}

// GetAccountInfo: Returns the account info.
func (r *RelyingpartyService) GetAccountInfo(identitytoolkitrelyingpartygetaccountinforequest *IdentitytoolkitRelyingpartyGetAccountInfoRequest) *RelyingpartyGetAccountInfoCall {
	c := &RelyingpartyGetAccountInfoCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartygetaccountinforequest = identitytoolkitrelyingpartygetaccountinforequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyGetAccountInfoCall) QuotaUser(quotaUser string) *RelyingpartyGetAccountInfoCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyGetAccountInfoCall) UserIP(userIP string) *RelyingpartyGetAccountInfoCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyGetAccountInfoCall) Fields(s ...googleapi.Field) *RelyingpartyGetAccountInfoCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyGetAccountInfoCall) Context(ctx context.Context) *RelyingpartyGetAccountInfoCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyGetAccountInfoCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartygetaccountinforequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "getAccountInfo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.getAccountInfo" call.
// Exactly one of *GetAccountInfoResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetAccountInfoResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyGetAccountInfoCall) Do() (*GetAccountInfoResponse, error) {
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
	ret := &GetAccountInfoResponse{
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
	//   "description": "Returns the account info.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.getAccountInfo",
	//   "path": "getAccountInfo",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyGetAccountInfoRequest"
	//   },
	//   "response": {
	//     "$ref": "GetAccountInfoResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.getOobConfirmationCode":

type RelyingpartyGetOobConfirmationCodeCall struct {
	s            *Service
	relyingparty *Relyingparty
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// GetOobConfirmationCode: Get a code for user action confirmation.
func (r *RelyingpartyService) GetOobConfirmationCode(relyingparty *Relyingparty) *RelyingpartyGetOobConfirmationCodeCall {
	c := &RelyingpartyGetOobConfirmationCodeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.relyingparty = relyingparty
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyGetOobConfirmationCodeCall) QuotaUser(quotaUser string) *RelyingpartyGetOobConfirmationCodeCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyGetOobConfirmationCodeCall) UserIP(userIP string) *RelyingpartyGetOobConfirmationCodeCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyGetOobConfirmationCodeCall) Fields(s ...googleapi.Field) *RelyingpartyGetOobConfirmationCodeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyGetOobConfirmationCodeCall) Context(ctx context.Context) *RelyingpartyGetOobConfirmationCodeCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyGetOobConfirmationCodeCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.relyingparty)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "getOobConfirmationCode")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.getOobConfirmationCode" call.
// Exactly one of *GetOobConfirmationCodeResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *GetOobConfirmationCodeResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyGetOobConfirmationCodeCall) Do() (*GetOobConfirmationCodeResponse, error) {
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
	ret := &GetOobConfirmationCodeResponse{
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
	//   "description": "Get a code for user action confirmation.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.getOobConfirmationCode",
	//   "path": "getOobConfirmationCode",
	//   "request": {
	//     "$ref": "Relyingparty"
	//   },
	//   "response": {
	//     "$ref": "GetOobConfirmationCodeResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.getProjectConfig":

type RelyingpartyGetProjectConfigCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// GetProjectConfig: Get project configuration.
func (r *RelyingpartyService) GetProjectConfig() *RelyingpartyGetProjectConfigCall {
	c := &RelyingpartyGetProjectConfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyGetProjectConfigCall) QuotaUser(quotaUser string) *RelyingpartyGetProjectConfigCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyGetProjectConfigCall) UserIP(userIP string) *RelyingpartyGetProjectConfigCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyGetProjectConfigCall) Fields(s ...googleapi.Field) *RelyingpartyGetProjectConfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RelyingpartyGetProjectConfigCall) IfNoneMatch(entityTag string) *RelyingpartyGetProjectConfigCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyGetProjectConfigCall) Context(ctx context.Context) *RelyingpartyGetProjectConfigCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyGetProjectConfigCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "getProjectConfig")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.getProjectConfig" call.
// Exactly one of *IdentitytoolkitRelyingpartyGetProjectConfigResponse
// or error will be non-nil. Any non-2xx status code is an error.
// Response headers are in either
// *IdentitytoolkitRelyingpartyGetProjectConfigResponse.ServerResponse.He
// ader or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *RelyingpartyGetProjectConfigCall) Do() (*IdentitytoolkitRelyingpartyGetProjectConfigResponse, error) {
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
	ret := &IdentitytoolkitRelyingpartyGetProjectConfigResponse{
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
	//   "description": "Get project configuration.",
	//   "httpMethod": "GET",
	//   "id": "identitytoolkit.relyingparty.getProjectConfig",
	//   "path": "getProjectConfig",
	//   "response": {
	//     "$ref": "IdentitytoolkitRelyingpartyGetProjectConfigResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.getPublicKeys":

type RelyingpartyGetPublicKeysCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// GetPublicKeys: Get token signing public key.
func (r *RelyingpartyService) GetPublicKeys() *RelyingpartyGetPublicKeysCall {
	c := &RelyingpartyGetPublicKeysCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyGetPublicKeysCall) QuotaUser(quotaUser string) *RelyingpartyGetPublicKeysCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyGetPublicKeysCall) UserIP(userIP string) *RelyingpartyGetPublicKeysCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyGetPublicKeysCall) Fields(s ...googleapi.Field) *RelyingpartyGetPublicKeysCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RelyingpartyGetPublicKeysCall) IfNoneMatch(entityTag string) *RelyingpartyGetPublicKeysCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyGetPublicKeysCall) Context(ctx context.Context) *RelyingpartyGetPublicKeysCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyGetPublicKeysCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "publicKeys")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.getPublicKeys" call.
func (c *RelyingpartyGetPublicKeysCall) Do() (map[string]string, error) {
	res, err := c.doRequest("json")
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret map[string]string
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get token signing public key.",
	//   "httpMethod": "GET",
	//   "id": "identitytoolkit.relyingparty.getPublicKeys",
	//   "path": "publicKeys",
	//   "response": {
	//     "$ref": "IdentitytoolkitRelyingpartyGetPublicKeysResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.getRecaptchaParam":

type RelyingpartyGetRecaptchaParamCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// GetRecaptchaParam: Get recaptcha secure param.
func (r *RelyingpartyService) GetRecaptchaParam() *RelyingpartyGetRecaptchaParamCall {
	c := &RelyingpartyGetRecaptchaParamCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyGetRecaptchaParamCall) QuotaUser(quotaUser string) *RelyingpartyGetRecaptchaParamCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyGetRecaptchaParamCall) UserIP(userIP string) *RelyingpartyGetRecaptchaParamCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyGetRecaptchaParamCall) Fields(s ...googleapi.Field) *RelyingpartyGetRecaptchaParamCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RelyingpartyGetRecaptchaParamCall) IfNoneMatch(entityTag string) *RelyingpartyGetRecaptchaParamCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyGetRecaptchaParamCall) Context(ctx context.Context) *RelyingpartyGetRecaptchaParamCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyGetRecaptchaParamCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "getRecaptchaParam")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.getRecaptchaParam" call.
// Exactly one of *GetRecaptchaParamResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GetRecaptchaParamResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyGetRecaptchaParamCall) Do() (*GetRecaptchaParamResponse, error) {
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
	ret := &GetRecaptchaParamResponse{
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
	//   "description": "Get recaptcha secure param.",
	//   "httpMethod": "GET",
	//   "id": "identitytoolkit.relyingparty.getRecaptchaParam",
	//   "path": "getRecaptchaParam",
	//   "response": {
	//     "$ref": "GetRecaptchaParamResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.resetPassword":

type RelyingpartyResetPasswordCall struct {
	s                                               *Service
	identitytoolkitrelyingpartyresetpasswordrequest *IdentitytoolkitRelyingpartyResetPasswordRequest
	urlParams_                                      gensupport.URLParams
	ctx_                                            context.Context
}

// ResetPassword: Reset password for a user.
func (r *RelyingpartyService) ResetPassword(identitytoolkitrelyingpartyresetpasswordrequest *IdentitytoolkitRelyingpartyResetPasswordRequest) *RelyingpartyResetPasswordCall {
	c := &RelyingpartyResetPasswordCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartyresetpasswordrequest = identitytoolkitrelyingpartyresetpasswordrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyResetPasswordCall) QuotaUser(quotaUser string) *RelyingpartyResetPasswordCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyResetPasswordCall) UserIP(userIP string) *RelyingpartyResetPasswordCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyResetPasswordCall) Fields(s ...googleapi.Field) *RelyingpartyResetPasswordCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyResetPasswordCall) Context(ctx context.Context) *RelyingpartyResetPasswordCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyResetPasswordCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartyresetpasswordrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "resetPassword")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.resetPassword" call.
// Exactly one of *ResetPasswordResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ResetPasswordResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyResetPasswordCall) Do() (*ResetPasswordResponse, error) {
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
	ret := &ResetPasswordResponse{
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
	//   "description": "Reset password for a user.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.resetPassword",
	//   "path": "resetPassword",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyResetPasswordRequest"
	//   },
	//   "response": {
	//     "$ref": "ResetPasswordResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.setAccountInfo":

type RelyingpartySetAccountInfoCall struct {
	s                                                *Service
	identitytoolkitrelyingpartysetaccountinforequest *IdentitytoolkitRelyingpartySetAccountInfoRequest
	urlParams_                                       gensupport.URLParams
	ctx_                                             context.Context
}

// SetAccountInfo: Set account info for a user.
func (r *RelyingpartyService) SetAccountInfo(identitytoolkitrelyingpartysetaccountinforequest *IdentitytoolkitRelyingpartySetAccountInfoRequest) *RelyingpartySetAccountInfoCall {
	c := &RelyingpartySetAccountInfoCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartysetaccountinforequest = identitytoolkitrelyingpartysetaccountinforequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartySetAccountInfoCall) QuotaUser(quotaUser string) *RelyingpartySetAccountInfoCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartySetAccountInfoCall) UserIP(userIP string) *RelyingpartySetAccountInfoCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartySetAccountInfoCall) Fields(s ...googleapi.Field) *RelyingpartySetAccountInfoCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartySetAccountInfoCall) Context(ctx context.Context) *RelyingpartySetAccountInfoCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartySetAccountInfoCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartysetaccountinforequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "setAccountInfo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.setAccountInfo" call.
// Exactly one of *SetAccountInfoResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SetAccountInfoResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartySetAccountInfoCall) Do() (*SetAccountInfoResponse, error) {
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
	ret := &SetAccountInfoResponse{
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
	//   "description": "Set account info for a user.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.setAccountInfo",
	//   "path": "setAccountInfo",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartySetAccountInfoRequest"
	//   },
	//   "response": {
	//     "$ref": "SetAccountInfoResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.uploadAccount":

type RelyingpartyUploadAccountCall struct {
	s                                               *Service
	identitytoolkitrelyingpartyuploadaccountrequest *IdentitytoolkitRelyingpartyUploadAccountRequest
	urlParams_                                      gensupport.URLParams
	ctx_                                            context.Context
}

// UploadAccount: Batch upload existing user accounts.
func (r *RelyingpartyService) UploadAccount(identitytoolkitrelyingpartyuploadaccountrequest *IdentitytoolkitRelyingpartyUploadAccountRequest) *RelyingpartyUploadAccountCall {
	c := &RelyingpartyUploadAccountCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartyuploadaccountrequest = identitytoolkitrelyingpartyuploadaccountrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyUploadAccountCall) QuotaUser(quotaUser string) *RelyingpartyUploadAccountCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyUploadAccountCall) UserIP(userIP string) *RelyingpartyUploadAccountCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyUploadAccountCall) Fields(s ...googleapi.Field) *RelyingpartyUploadAccountCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyUploadAccountCall) Context(ctx context.Context) *RelyingpartyUploadAccountCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyUploadAccountCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartyuploadaccountrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "uploadAccount")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.uploadAccount" call.
// Exactly one of *UploadAccountResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *UploadAccountResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyUploadAccountCall) Do() (*UploadAccountResponse, error) {
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
	ret := &UploadAccountResponse{
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
	//   "description": "Batch upload existing user accounts.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.uploadAccount",
	//   "path": "uploadAccount",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyUploadAccountRequest"
	//   },
	//   "response": {
	//     "$ref": "UploadAccountResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.verifyAssertion":

type RelyingpartyVerifyAssertionCall struct {
	s                                                 *Service
	identitytoolkitrelyingpartyverifyassertionrequest *IdentitytoolkitRelyingpartyVerifyAssertionRequest
	urlParams_                                        gensupport.URLParams
	ctx_                                              context.Context
}

// VerifyAssertion: Verifies the assertion returned by the IdP.
func (r *RelyingpartyService) VerifyAssertion(identitytoolkitrelyingpartyverifyassertionrequest *IdentitytoolkitRelyingpartyVerifyAssertionRequest) *RelyingpartyVerifyAssertionCall {
	c := &RelyingpartyVerifyAssertionCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartyverifyassertionrequest = identitytoolkitrelyingpartyverifyassertionrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyVerifyAssertionCall) QuotaUser(quotaUser string) *RelyingpartyVerifyAssertionCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyVerifyAssertionCall) UserIP(userIP string) *RelyingpartyVerifyAssertionCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyVerifyAssertionCall) Fields(s ...googleapi.Field) *RelyingpartyVerifyAssertionCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyVerifyAssertionCall) Context(ctx context.Context) *RelyingpartyVerifyAssertionCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyVerifyAssertionCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartyverifyassertionrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "verifyAssertion")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.verifyAssertion" call.
// Exactly one of *VerifyAssertionResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *VerifyAssertionResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyVerifyAssertionCall) Do() (*VerifyAssertionResponse, error) {
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
	ret := &VerifyAssertionResponse{
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
	//   "description": "Verifies the assertion returned by the IdP.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.verifyAssertion",
	//   "path": "verifyAssertion",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyVerifyAssertionRequest"
	//   },
	//   "response": {
	//     "$ref": "VerifyAssertionResponse"
	//   }
	// }

}

// method id "identitytoolkit.relyingparty.verifyPassword":

type RelyingpartyVerifyPasswordCall struct {
	s                                                *Service
	identitytoolkitrelyingpartyverifypasswordrequest *IdentitytoolkitRelyingpartyVerifyPasswordRequest
	urlParams_                                       gensupport.URLParams
	ctx_                                             context.Context
}

// VerifyPassword: Verifies the user entered password.
func (r *RelyingpartyService) VerifyPassword(identitytoolkitrelyingpartyverifypasswordrequest *IdentitytoolkitRelyingpartyVerifyPasswordRequest) *RelyingpartyVerifyPasswordCall {
	c := &RelyingpartyVerifyPasswordCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.identitytoolkitrelyingpartyverifypasswordrequest = identitytoolkitrelyingpartyverifypasswordrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *RelyingpartyVerifyPasswordCall) QuotaUser(quotaUser string) *RelyingpartyVerifyPasswordCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *RelyingpartyVerifyPasswordCall) UserIP(userIP string) *RelyingpartyVerifyPasswordCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RelyingpartyVerifyPasswordCall) Fields(s ...googleapi.Field) *RelyingpartyVerifyPasswordCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RelyingpartyVerifyPasswordCall) Context(ctx context.Context) *RelyingpartyVerifyPasswordCall {
	c.ctx_ = ctx
	return c
}

func (c *RelyingpartyVerifyPasswordCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.identitytoolkitrelyingpartyverifypasswordrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "verifyPassword")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "identitytoolkit.relyingparty.verifyPassword" call.
// Exactly one of *VerifyPasswordResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *VerifyPasswordResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RelyingpartyVerifyPasswordCall) Do() (*VerifyPasswordResponse, error) {
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
	ret := &VerifyPasswordResponse{
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
	//   "description": "Verifies the user entered password.",
	//   "httpMethod": "POST",
	//   "id": "identitytoolkit.relyingparty.verifyPassword",
	//   "path": "verifyPassword",
	//   "request": {
	//     "$ref": "IdentitytoolkitRelyingpartyVerifyPasswordRequest"
	//   },
	//   "response": {
	//     "$ref": "VerifyPasswordResponse"
	//   }
	// }

}
