// Package androidenterprise provides access to the Google Play EMM API.
//
// See https://developers.google.com/android/work/play/emm-api
//
// Usage example:
//
//   import "google.golang.org/api/androidenterprise/v1"
//   ...
//   androidenterpriseService, err := androidenterprise.New(oauthHttpClient)
package androidenterprise // import "google.golang.org/api/androidenterprise/v1"

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

const apiId = "androidenterprise:v1"
const apiName = "androidenterprise"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/androidenterprise/v1/"

// OAuth2 scopes used by this API.
const (
	// Manage corporate Android devices
	AndroidenterpriseScope = "https://www.googleapis.com/auth/androidenterprise"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Devices = NewDevicesService(s)
	s.Enterprises = NewEnterprisesService(s)
	s.Entitlements = NewEntitlementsService(s)
	s.Grouplicenses = NewGrouplicensesService(s)
	s.Grouplicenseusers = NewGrouplicenseusersService(s)
	s.Installs = NewInstallsService(s)
	s.Managedconfigurationsfordevice = NewManagedconfigurationsfordeviceService(s)
	s.Managedconfigurationsforuser = NewManagedconfigurationsforuserService(s)
	s.Permissions = NewPermissionsService(s)
	s.Products = NewProductsService(s)
	s.Serviceaccountkeys = NewServiceaccountkeysService(s)
	s.Storelayoutclusters = NewStorelayoutclustersService(s)
	s.Storelayoutpages = NewStorelayoutpagesService(s)
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Devices *DevicesService

	Enterprises *EnterprisesService

	Entitlements *EntitlementsService

	Grouplicenses *GrouplicensesService

	Grouplicenseusers *GrouplicenseusersService

	Installs *InstallsService

	Managedconfigurationsfordevice *ManagedconfigurationsfordeviceService

	Managedconfigurationsforuser *ManagedconfigurationsforuserService

	Permissions *PermissionsService

	Products *ProductsService

	Serviceaccountkeys *ServiceaccountkeysService

	Storelayoutclusters *StorelayoutclustersService

	Storelayoutpages *StorelayoutpagesService

	Users *UsersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewDevicesService(s *Service) *DevicesService {
	rs := &DevicesService{s: s}
	return rs
}

type DevicesService struct {
	s *Service
}

func NewEnterprisesService(s *Service) *EnterprisesService {
	rs := &EnterprisesService{s: s}
	return rs
}

type EnterprisesService struct {
	s *Service
}

func NewEntitlementsService(s *Service) *EntitlementsService {
	rs := &EntitlementsService{s: s}
	return rs
}

type EntitlementsService struct {
	s *Service
}

func NewGrouplicensesService(s *Service) *GrouplicensesService {
	rs := &GrouplicensesService{s: s}
	return rs
}

type GrouplicensesService struct {
	s *Service
}

func NewGrouplicenseusersService(s *Service) *GrouplicenseusersService {
	rs := &GrouplicenseusersService{s: s}
	return rs
}

type GrouplicenseusersService struct {
	s *Service
}

func NewInstallsService(s *Service) *InstallsService {
	rs := &InstallsService{s: s}
	return rs
}

type InstallsService struct {
	s *Service
}

func NewManagedconfigurationsfordeviceService(s *Service) *ManagedconfigurationsfordeviceService {
	rs := &ManagedconfigurationsfordeviceService{s: s}
	return rs
}

type ManagedconfigurationsfordeviceService struct {
	s *Service
}

func NewManagedconfigurationsforuserService(s *Service) *ManagedconfigurationsforuserService {
	rs := &ManagedconfigurationsforuserService{s: s}
	return rs
}

type ManagedconfigurationsforuserService struct {
	s *Service
}

func NewPermissionsService(s *Service) *PermissionsService {
	rs := &PermissionsService{s: s}
	return rs
}

type PermissionsService struct {
	s *Service
}

func NewProductsService(s *Service) *ProductsService {
	rs := &ProductsService{s: s}
	return rs
}

type ProductsService struct {
	s *Service
}

func NewServiceaccountkeysService(s *Service) *ServiceaccountkeysService {
	rs := &ServiceaccountkeysService{s: s}
	return rs
}

type ServiceaccountkeysService struct {
	s *Service
}

func NewStorelayoutclustersService(s *Service) *StorelayoutclustersService {
	rs := &StorelayoutclustersService{s: s}
	return rs
}

type StorelayoutclustersService struct {
	s *Service
}

func NewStorelayoutpagesService(s *Service) *StorelayoutpagesService {
	rs := &StorelayoutpagesService{s: s}
	return rs
}

type StorelayoutpagesService struct {
	s *Service
}

func NewUsersService(s *Service) *UsersService {
	rs := &UsersService{s: s}
	return rs
}

type UsersService struct {
	s *Service
}

// Administrator: This represents an enterprise admin who can manage the
// enterprise in the managed Google Play store.
type Administrator struct {
	// Email: The admin's email address.
	Email string `json:"email,omitempty"`

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

func (s *Administrator) MarshalJSON() ([]byte, error) {
	type noMethod Administrator
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AdministratorWebToken: A token authorizing an admin to access an
// iframe.
type AdministratorWebToken struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#administratorWebToken".
	Kind string `json:"kind,omitempty"`

	// Token: An opaque token to be passed to the Play front-end to generate
	// an iframe.
	Token string `json:"token,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AdministratorWebToken) MarshalJSON() ([]byte, error) {
	type noMethod AdministratorWebToken
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AdministratorWebTokenSpec: Specification for a token used to generate
// iframes. The token specifies what data the admin is allowed to modify
// and the URI the iframe is allowed to communiate with.
type AdministratorWebTokenSpec struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#administratorWebTokenSpec".
	Kind string `json:"kind,omitempty"`

	// Parent: The URI of the parent frame hosting the iframe. To prevent
	// XSS, the iframe may not be hosted at other URIs. This URI must be
	// https.
	Parent string `json:"parent,omitempty"`

	// Permission: The list of permissions the admin is granted within the
	// iframe. The admin will only be allowed to view an iframe if they have
	// all of the permissions associated with it. The only valid value is
	// "approveApps" that will allow the admin to access the iframe in
	// "approve" mode.
	Permission []string `json:"permission,omitempty"`

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

func (s *AdministratorWebTokenSpec) MarshalJSON() ([]byte, error) {
	type noMethod AdministratorWebTokenSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AndroidDevicePolicyConfig: The Android Device Policy configuration of
// an enterprise.
type AndroidDevicePolicyConfig struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#androidDevicePolicyConfig".
	Kind string `json:"kind,omitempty"`

	// State: The state of Android Device Policy. "enabled" indicates that
	// Android Device Policy is enabled for the enterprise and the EMM is
	// allowed to manage devices with Android Device Policy, while
	// "disabled" means that it cannot.
	State string `json:"state,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AndroidDevicePolicyConfig) MarshalJSON() ([]byte, error) {
	type noMethod AndroidDevicePolicyConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppRestrictionsSchema: Represents the list of app restrictions
// available to be pre-configured for the product.
type AppRestrictionsSchema struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#appRestrictionsSchema".
	Kind string `json:"kind,omitempty"`

	// Restrictions: The set of restrictions that make up this schema.
	Restrictions []*AppRestrictionsSchemaRestriction `json:"restrictions,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppRestrictionsSchema) MarshalJSON() ([]byte, error) {
	type noMethod AppRestrictionsSchema
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppRestrictionsSchemaChangeEvent: An event generated when a new app
// version is uploaded to Google Play and its app restrictions schema
// changed. To fetch the app restrictions schema for an app, use
// Products.getAppRestrictionsSchema on the EMM API.
type AppRestrictionsSchemaChangeEvent struct {
	// ProductId: The id of the product (e.g. "app:com.google.android.gm")
	// for which the app restriction schema changed. This field will always
	// be present.
	ProductId string `json:"productId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProductId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProductId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppRestrictionsSchemaChangeEvent) MarshalJSON() ([]byte, error) {
	type noMethod AppRestrictionsSchemaChangeEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppRestrictionsSchemaRestriction: A restriction in the App
// Restriction Schema represents a piece of configuration that may be
// pre-applied.
type AppRestrictionsSchemaRestriction struct {
	// DefaultValue: The default value of the restriction. bundle and
	// bundleArray restrictions never have a default value.
	DefaultValue *AppRestrictionsSchemaRestrictionRestrictionValue `json:"defaultValue,omitempty"`

	// Description: A longer description of the restriction, giving more
	// detail of what it affects.
	Description string `json:"description,omitempty"`

	// Entry: For choice or multiselect restrictions, the list of possible
	// entries' human-readable names.
	Entry []string `json:"entry,omitempty"`

	// EntryValue: For choice or multiselect restrictions, the list of
	// possible entries' machine-readable values. These values should be
	// used in the configuration, either as a single string value for a
	// choice restriction or in a stringArray for a multiselect restriction.
	EntryValue []string `json:"entryValue,omitempty"`

	// Key: The unique key that the product uses to identify the
	// restriction, e.g. "com.google.android.gm.fieldname".
	Key string `json:"key,omitempty"`

	// NestedRestriction: For bundle or bundleArray restrictions, the list
	// of nested restrictions. A bundle restriction is always nested within
	// a bundleArray restriction, and a bundleArray restriction is at most
	// two levels deep.
	NestedRestriction []*AppRestrictionsSchemaRestriction `json:"nestedRestriction,omitempty"`

	// RestrictionType: The type of the restriction.
	RestrictionType string `json:"restrictionType,omitempty"`

	// Title: The name of the restriction.
	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DefaultValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DefaultValue") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppRestrictionsSchemaRestriction) MarshalJSON() ([]byte, error) {
	type noMethod AppRestrictionsSchemaRestriction
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppRestrictionsSchemaRestrictionRestrictionValue: A typed value for
// the restriction.
type AppRestrictionsSchemaRestrictionRestrictionValue struct {
	// Type: The type of the value being provided.
	Type string `json:"type,omitempty"`

	// ValueBool: The boolean value - this will only be present if type is
	// bool.
	ValueBool bool `json:"valueBool,omitempty"`

	// ValueInteger: The integer value - this will only be present if type
	// is integer.
	ValueInteger int64 `json:"valueInteger,omitempty"`

	// ValueMultiselect: The list of string values - this will only be
	// present if type is multiselect.
	ValueMultiselect []string `json:"valueMultiselect,omitempty"`

	// ValueString: The string value - this will be present for types
	// string, choice and hidden.
	ValueString string `json:"valueString,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Type") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Type") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppRestrictionsSchemaRestrictionRestrictionValue) MarshalJSON() ([]byte, error) {
	type noMethod AppRestrictionsSchemaRestrictionRestrictionValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppUpdateEvent: An event generated when a new version of an app is
// uploaded to Google Play. Notifications are sent for new public
// versions only: alpha, beta, or canary versions do not generate this
// event. To fetch up-to-date version history for an app, use
// Products.Get on the EMM API.
type AppUpdateEvent struct {
	// ProductId: The id of the product (e.g. "app:com.google.android.gm")
	// that was updated. This field will always be present.
	ProductId string `json:"productId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProductId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProductId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppUpdateEvent) MarshalJSON() ([]byte, error) {
	type noMethod AppUpdateEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppVersion: This represents a single version of the app.
type AppVersion struct {
	// VersionCode: Unique increasing identifier for the app version.
	VersionCode int64 `json:"versionCode,omitempty"`

	// VersionString: The string used in the Play store by the app developer
	// to identify the version. The string is not necessarily unique or
	// localized (for example, the string could be "1.4").
	VersionString string `json:"versionString,omitempty"`

	// ForceSendFields is a list of field names (e.g. "VersionCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "VersionCode") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppVersion) MarshalJSON() ([]byte, error) {
	type noMethod AppVersion
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ApprovalUrlInfo: Information on an approval URL.
type ApprovalUrlInfo struct {
	// ApprovalUrl: A URL that displays a product's permissions and that can
	// also be used to approve the product with the Products.approve call.
	ApprovalUrl string `json:"approvalUrl,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#approvalUrlInfo".
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApprovalUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApprovalUrl") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ApprovalUrlInfo) MarshalJSON() ([]byte, error) {
	type noMethod ApprovalUrlInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthenticationToken: An AuthenticationToken is used by the EMM's
// device policy client on a device to provision the given EMM-managed
// user on that device.
type AuthenticationToken struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#authenticationToken".
	Kind string `json:"kind,omitempty"`

	// Token: The authentication token to be passed to the device policy
	// client on the device where it can be used to provision the account
	// for which this token was generated.
	Token string `json:"token,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuthenticationToken) MarshalJSON() ([]byte, error) {
	type noMethod AuthenticationToken
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Device: A Devices resource represents a mobile device managed by the
// EMM and belonging to a specific enterprise user.
//
// This collection cannot be modified via the API. It is automatically
// populated as devices are set up to be managed.
type Device struct {
	// AndroidId: The Google Play Services Android ID for the device encoded
	// as a lowercase hex string. For example, "123456789abcdef0".
	AndroidId string `json:"androidId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#device".
	Kind string `json:"kind,omitempty"`

	// ManagementType: Identifies the extent to which the device is
	// controlled by a managed Google Play EMM in various deployment
	// configurations.
	//
	// Possible values include:
	// - "managedDevice", a device that has the EMM's device policy
	// controller (DPC) as the device owner.
	// - "managedProfile", a device that has a profile managed by the DPC
	// (DPC is profile owner) in addition to a separate, personal profile
	// that is unavailable to the DPC.
	// - "containerApp", no longer used (deprecated).
	// - "unmanagedProfile", a device that has been allowed (by the domain's
	// admin, using the Admin Console to enable the privilege) to use
	// managed Google Play, but the profile is itself not owned by a DPC.
	ManagementType string `json:"managementType,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AndroidId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AndroidId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Device) MarshalJSON() ([]byte, error) {
	type noMethod Device
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeviceState: The state of a user's device, as accessed by the
// getState and setState methods on device resources.
type DeviceState struct {
	// AccountState: The state of the Google account on the device.
	// "enabled" indicates that the Google account on the device can be used
	// to access Google services (including Google Play), while "disabled"
	// means that it cannot. A new device is initially in the "disabled"
	// state.
	AccountState string `json:"accountState,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#deviceState".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountState") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountState") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeviceState) MarshalJSON() ([]byte, error) {
	type noMethod DeviceState
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DevicesListResponse: The device resources for the user.
type DevicesListResponse struct {
	// Device: A managed device.
	Device []*Device `json:"device,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#devicesListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Device") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Device") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DevicesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DevicesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Enterprise: An Enterprises resource represents the binding between an
// EMM and a specific organization. That binding can be instantiated in
// one of two different ways using this API as follows:
// - For Google managed domain customers, the process involves using
// Enterprises.enroll and Enterprises.setAccount (in conjunction with
// artifacts obtained from the Admin console and the Google API Console)
// and submitted to the EMM through a more-or-less manual process.
// - For managed Google Play Accounts customers, the process involves
// using Enterprises.generateSignupUrl and Enterprises.completeSignup in
// conjunction with the managed Google Play sign-up UI (Google-provided
// mechanism) to create the binding without manual steps. As an EMM, you
// can support either or both approaches in your EMM console. See Create
// an Enterprise for details.
type Enterprise struct {
	// Administrator: Admins of the enterprise. This is only supported for
	// enterprises created via the EMM-initiated flow.
	Administrator []*Administrator `json:"administrator,omitempty"`

	// Id: The unique ID for the enterprise.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#enterprise".
	Kind string `json:"kind,omitempty"`

	// Name: The name of the enterprise, for example, "Example, Inc".
	Name string `json:"name,omitempty"`

	// PrimaryDomain: The enterprise's primary domain, such as
	// "example.com".
	PrimaryDomain string `json:"primaryDomain,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Administrator") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Administrator") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Enterprise) MarshalJSON() ([]byte, error) {
	type noMethod Enterprise
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EnterpriseAccount: A service account that can be used to authenticate
// as the enterprise to API calls that require such authentication.
type EnterpriseAccount struct {
	// AccountEmail: The email address of the service account.
	AccountEmail string `json:"accountEmail,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#enterpriseAccount".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountEmail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountEmail") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EnterpriseAccount) MarshalJSON() ([]byte, error) {
	type noMethod EnterpriseAccount
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EnterprisesListResponse: The matching enterprise resources.
type EnterprisesListResponse struct {
	// Enterprise: An enterprise.
	Enterprise []*Enterprise `json:"enterprise,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#enterprisesListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Enterprise") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Enterprise") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EnterprisesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod EnterprisesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type EnterprisesSendTestPushNotificationResponse struct {
	// MessageId: The message ID of the test push notification that was
	// sent.
	MessageId string `json:"messageId,omitempty"`

	// TopicName: The name of the Cloud Pub/Sub topic to which notifications
	// for this enterprise's enrolled account will be sent.
	TopicName string `json:"topicName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "MessageId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MessageId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EnterprisesSendTestPushNotificationResponse) MarshalJSON() ([]byte, error) {
	type noMethod EnterprisesSendTestPushNotificationResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Entitlement: The presence of an Entitlements resource indicates that
// a user has the right to use a particular app. Entitlements are user
// specific, not device specific. This allows a user with an entitlement
// to an app to install the app on all their devices. It's also possible
// for a user to hold an entitlement to an app without installing the
// app on any device.
//
// The API can be used to create an entitlement. As an option, you can
// also use the API to trigger the installation of an app on all a
// user's managed devices at the same time the entitlement is
// created.
//
// If the app is free, creating the entitlement also creates a group
// license for that app. For paid apps, creating the entitlement
// consumes one license, and that license remains consumed until the
// entitlement is removed. If the enterprise hasn't purchased enough
// licenses, then no entitlement is created and the installation fails.
// An entitlement is also not created for an app if the app requires
// permissions that the enterprise hasn't accepted.
//
// If an entitlement is deleted, the app may be uninstalled from a
// user's device. As a best practice, uninstall the app by calling
// Installs.delete() before deleting the entitlement.
//
// Entitlements for apps that a user pays for on an unmanaged profile
// have "userPurchase" as the entitlement reason. These entitlements
// cannot be removed via the API.
type Entitlement struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#entitlement".
	Kind string `json:"kind,omitempty"`

	// ProductId: The ID of the product that the entitlement is for. For
	// example, "app:com.google.android.gm".
	ProductId string `json:"productId,omitempty"`

	// Reason: The reason for the entitlement. For example, "free" for free
	// apps. This property is temporary: it will be replaced by the
	// acquisition kind field of group licenses.
	Reason string `json:"reason,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Entitlement) MarshalJSON() ([]byte, error) {
	type noMethod Entitlement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EntitlementsListResponse: The entitlement resources for the user.
type EntitlementsListResponse struct {
	// Entitlement: An entitlement of a user to a product (e.g. an app). For
	// example, a free app that they have installed, or a paid app that they
	// have been allocated a license to.
	Entitlement []*Entitlement `json:"entitlement,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#entitlementsListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entitlement") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entitlement") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EntitlementsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod EntitlementsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GroupLicense: Group license objects allow you to keep track of
// licenses (called entitlements) for both free and paid apps. For a
// free app, a group license is created when an enterprise admin first
// approves the product in Google Play or when the first entitlement for
// the product is created for a user via the API. For a paid app, a
// group license object is only created when an enterprise admin
// purchases the product in Google Play for the first time.
//
// Use the API to query group licenses. A Grouplicenses resource
// includes the total number of licenses purchased (paid apps only) and
// the total number of licenses currently in use. Iyn other words, the
// total number of Entitlements that exist for the product.
//
// Only one group license object is created per product and group
// license objects are never deleted. If a product is unapproved, its
// group license remains. This allows enterprise admins to keep track of
// any remaining entitlements for the product.
type GroupLicense struct {
	// AcquisitionKind: How this group license was acquired. "bulkPurchase"
	// means that this Grouplicenses resource was created because the
	// enterprise purchased licenses for this product; otherwise, the value
	// is "free" (for free products).
	AcquisitionKind string `json:"acquisitionKind,omitempty"`

	// Approval: Whether the product to which this group license relates is
	// currently approved by the enterprise. Products are approved when a
	// group license is first created, but this approval may be revoked by
	// an enterprise admin via Google Play. Unapproved products will not be
	// visible to end users in collections, and new entitlements to them
	// should not normally be created.
	Approval string `json:"approval,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#groupLicense".
	Kind string `json:"kind,omitempty"`

	// NumProvisioned: The total number of provisioned licenses for this
	// product. Returned by read operations, but ignored in write
	// operations.
	NumProvisioned int64 `json:"numProvisioned,omitempty"`

	// NumPurchased: The number of purchased licenses (possibly in multiple
	// purchases). If this field is omitted, then there is no limit on the
	// number of licenses that can be provisioned (for example, if the
	// acquisition kind is "free").
	NumPurchased int64 `json:"numPurchased,omitempty"`

	// Permissions: The permission approval status of the product. This
	// field is only set if the product is approved. Possible states are:
	// - "currentApproved", the current set of permissions is approved, but
	// additional permissions will require the administrator to reapprove
	// the product (If the product was approved without specifying the
	// approved permissions setting, then this is the default behavior.),
	// - "needsReapproval", the product has unapproved permissions. No
	// additional product licenses can be assigned until the product is
	// reapproved,
	// - "allCurrentAndFutureApproved", the current permissions are approved
	// and any future permission updates will be automatically approved
	// without administrator review.
	Permissions string `json:"permissions,omitempty"`

	// ProductId: The ID of the product that the license is for. For
	// example, "app:com.google.android.gm".
	ProductId string `json:"productId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AcquisitionKind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AcquisitionKind") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GroupLicense) MarshalJSON() ([]byte, error) {
	type noMethod GroupLicense
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GroupLicenseUsersListResponse: The user resources for the group
// license.
type GroupLicenseUsersListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#groupLicenseUsersListResponse".
	Kind string `json:"kind,omitempty"`

	// User: A user of an enterprise.
	User []*User `json:"user,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GroupLicenseUsersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod GroupLicenseUsersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GroupLicensesListResponse: The grouplicense resources for the
// enterprise.
type GroupLicensesListResponse struct {
	// GroupLicense: A group license for a product approved for use in the
	// enterprise.
	GroupLicense []*GroupLicense `json:"groupLicense,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#groupLicensesListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "GroupLicense") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GroupLicense") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GroupLicensesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod GroupLicensesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Install: The existence of an Installs resource indicates that an app
// is installed on a particular device (or that an install is
// pending).
//
// The API can be used to create an install resource using the update
// method. This triggers the actual install of the app on the device. If
// the user does not already have an entitlement for the app, then an
// attempt is made to create one. If this fails (for example, because
// the app is not free and there is no available license), then the
// creation of the install fails.
//
// The API can also be used to update an installed app. If
// the update method is used on an existing install, then the app will
// be updated to the latest available version.
//
// Note that it is not possible to force the installation of a specific
// version of an app: the version code is read-only.
//
// If a user installs an app themselves (as permitted by the
// enterprise), then again an install resource and possibly an
// entitlement resource are automatically created.
//
// The API can also be used to delete an install resource, which
// triggers the removal of the app from the device. Note that deleting
// an install does not automatically remove the corresponding
// entitlement, even if there are no remaining installs. The install
// resource will also be deleted if the user uninstalls the app
// themselves.
type Install struct {
	// InstallState: Install state. The state "installPending" means that an
	// install request has recently been made and download to the device is
	// in progress. The state "installed" means that the app has been
	// installed. This field is read-only.
	InstallState string `json:"installState,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#install".
	Kind string `json:"kind,omitempty"`

	// ProductId: The ID of the product that the install is for. For
	// example, "app:com.google.android.gm".
	ProductId string `json:"productId,omitempty"`

	// VersionCode: The version of the installed product. Guaranteed to be
	// set only if the install state is "installed".
	VersionCode int64 `json:"versionCode,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "InstallState") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InstallState") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Install) MarshalJSON() ([]byte, error) {
	type noMethod Install
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstallFailureEvent: An event generated when an app installation
// failed on a device
type InstallFailureEvent struct {
	// DeviceId: The Android ID of the device. This field will always be
	// present.
	DeviceId string `json:"deviceId,omitempty"`

	// FailureDetails: Additional details on the failure if applicable.
	FailureDetails string `json:"failureDetails,omitempty"`

	// FailureReason: The reason for the installation failure. This field
	// will always be present.
	FailureReason string `json:"failureReason,omitempty"`

	// ProductId: The id of the product (e.g. "app:com.google.android.gm")
	// for which the install failure event occured. This field will always
	// be present.
	ProductId string `json:"productId,omitempty"`

	// UserId: The ID of the user. This field will always be present.
	UserId string `json:"userId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeviceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeviceId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstallFailureEvent) MarshalJSON() ([]byte, error) {
	type noMethod InstallFailureEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstallsListResponse: The install resources for the device.
type InstallsListResponse struct {
	// Install: An installation of an app for a user on a specific device.
	// The existence of an install implies that the user must have an
	// entitlement to the app.
	Install []*Install `json:"install,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#installsListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Install") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Install") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstallsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod InstallsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LocalizedText: A localized string with its locale.
type LocalizedText struct {
	// Locale: The BCP47 tag for a locale. (e.g. "en-US", "de").
	Locale string `json:"locale,omitempty"`

	// Text: The text localized in the associated locale.
	Text string `json:"text,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Locale") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Locale") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LocalizedText) MarshalJSON() ([]byte, error) {
	type noMethod LocalizedText
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedConfiguration: A managed configuration resource contains the
// set of managed properties that have been configured for an Android
// app. The app's developer would have defined configurable properties
// in the managed configurations schema.
type ManagedConfiguration struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#managedConfiguration".
	Kind string `json:"kind,omitempty"`

	// ManagedProperty: The set of managed properties for this
	// configuration.
	ManagedProperty []*ManagedProperty `json:"managedProperty,omitempty"`

	// ProductId: The ID of the product that the managed configuration is
	// for, e.g. "app:com.google.android.gm".
	ProductId string `json:"productId,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ManagedConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod ManagedConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedConfigurationsForDeviceListResponse: The managed configuration
// resources for the device.
type ManagedConfigurationsForDeviceListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string
	// "androidenterprise#managedConfigurationsForDeviceListResponse".
	Kind string `json:"kind,omitempty"`

	// ManagedConfigurationForDevice: A managed configuration for an app on
	// a specific device.
	ManagedConfigurationForDevice []*ManagedConfiguration `json:"managedConfigurationForDevice,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ManagedConfigurationsForDeviceListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ManagedConfigurationsForDeviceListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedConfigurationsForUserListResponse: The managed configuration
// resources for the user.
type ManagedConfigurationsForUserListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#managedConfigurationsForUserListResponse".
	Kind string `json:"kind,omitempty"`

	// ManagedConfigurationForUser: A managed configuration for an app for a
	// specific user.
	ManagedConfigurationForUser []*ManagedConfiguration `json:"managedConfigurationForUser,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ManagedConfigurationsForUserListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ManagedConfigurationsForUserListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedProperty: A managed property of a managed configuration. The
// property must match one of the properties in the app restrictions
// schema of the product. Exactly one of the value fields must be
// populated, and it must match the property's type in the app
// restrictions schema.
type ManagedProperty struct {
	// Key: The unique key that identifies the property.
	Key string `json:"key,omitempty"`

	// ValueBool: The boolean value - this will only be present if type of
	// the property is bool.
	ValueBool bool `json:"valueBool,omitempty"`

	// ValueBundle: The bundle of managed properties - this will only be
	// present if type of the property is bundle.
	ValueBundle *ManagedPropertyBundle `json:"valueBundle,omitempty"`

	// ValueBundleArray: The list of bundles of properties - this will only
	// be present if type of the property is bundle_array.
	ValueBundleArray []*ManagedPropertyBundle `json:"valueBundleArray,omitempty"`

	// ValueInteger: The integer value - this will only be present if type
	// of the property is integer.
	ValueInteger int64 `json:"valueInteger,omitempty"`

	// ValueString: The string value - this will only be present if type of
	// the property is string, choice or hidden.
	ValueString string `json:"valueString,omitempty"`

	// ValueStringArray: The list of string values - this will only be
	// present if type of the property is multiselect.
	ValueStringArray []string `json:"valueStringArray,omitempty"`

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

func (s *ManagedProperty) MarshalJSON() ([]byte, error) {
	type noMethod ManagedProperty
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedPropertyBundle: A bundle of managed properties.
type ManagedPropertyBundle struct {
	// ManagedProperty: The list of managed properties.
	ManagedProperty []*ManagedProperty `json:"managedProperty,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ManagedProperty") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ManagedProperty") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ManagedPropertyBundle) MarshalJSON() ([]byte, error) {
	type noMethod ManagedPropertyBundle
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NewDeviceEvent: An event generated when a new device is ready to be
// managed.
type NewDeviceEvent struct {
	// DeviceId: The Android ID of the device. This field will always be
	// present.
	DeviceId string `json:"deviceId,omitempty"`

	// ManagementType: Identifies the extent to which the device is
	// controlled by an Android EMM in various deployment
	// configurations.
	//
	// Possible values include:
	// - "managedDevice", a device where the DPC is set as device owner,
	// - "managedProfile", a device where the DPC is set as profile owner.
	ManagementType string `json:"managementType,omitempty"`

	// UserId: The ID of the user. This field will always be present.
	UserId string `json:"userId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeviceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeviceId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NewDeviceEvent) MarshalJSON() ([]byte, error) {
	type noMethod NewDeviceEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NewPermissionsEvent: An event generated when new permissions are
// added to an app.
type NewPermissionsEvent struct {
	// ApprovedPermissions: The set of permissions that the enterprise admin
	// has already approved for this application. Use Permissions.Get on the
	// EMM API to retrieve details about these permissions.
	ApprovedPermissions []string `json:"approvedPermissions,omitempty"`

	// ProductId: The id of the product (e.g. "app:com.google.android.gm")
	// for which new permissions were added. This field will always be
	// present.
	ProductId string `json:"productId,omitempty"`

	// RequestedPermissions: The set of permissions that the app is
	// currently requesting. Use Permissions.Get on the EMM API to retrieve
	// details about these permissions.
	RequestedPermissions []string `json:"requestedPermissions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApprovedPermissions")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApprovedPermissions") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *NewPermissionsEvent) MarshalJSON() ([]byte, error) {
	type noMethod NewPermissionsEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Notification: A notification of one event relating to an enterprise.
type Notification struct {
	// AppRestrictionsSchemaChangeEvent: Notifications about new app
	// restrictions schema changes.
	AppRestrictionsSchemaChangeEvent *AppRestrictionsSchemaChangeEvent `json:"appRestrictionsSchemaChangeEvent,omitempty"`

	// AppUpdateEvent: Notifications about app updates.
	AppUpdateEvent *AppUpdateEvent `json:"appUpdateEvent,omitempty"`

	// EnterpriseId: The ID of the enterprise for which the notification is
	// sent. This will always be present.
	EnterpriseId string `json:"enterpriseId,omitempty"`

	// InstallFailureEvent: Notifications about an app installation failure.
	InstallFailureEvent *InstallFailureEvent `json:"installFailureEvent,omitempty"`

	// NewDeviceEvent: Notifications about new devices.
	NewDeviceEvent *NewDeviceEvent `json:"newDeviceEvent,omitempty"`

	// NewPermissionsEvent: Notifications about new app permissions.
	NewPermissionsEvent *NewPermissionsEvent `json:"newPermissionsEvent,omitempty"`

	// NotificationType: Type of the notification.
	NotificationType string `json:"notificationType,omitempty"`

	// ProductApprovalEvent: Notifications about changes to a product's
	// approval status.
	ProductApprovalEvent *ProductApprovalEvent `json:"productApprovalEvent,omitempty"`

	// ProductAvailabilityChangeEvent: Notifications about product
	// availability changes.
	ProductAvailabilityChangeEvent *ProductAvailabilityChangeEvent `json:"productAvailabilityChangeEvent,omitempty"`

	// TimestampMillis: The time when the notification was published in
	// milliseconds since 1970-01-01T00:00:00Z. This will always be present.
	TimestampMillis int64 `json:"timestampMillis,omitempty,string"`

	// ForceSendFields is a list of field names (e.g.
	// "AppRestrictionsSchemaChangeEvent") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "AppRestrictionsSchemaChangeEvent") to include in API requests with
	// the JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Notification) MarshalJSON() ([]byte, error) {
	type noMethod Notification
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NotificationSet: A resource returned by the PullNotificationSet API,
// which contains a collection of notifications for enterprises
// associated with the service account authenticated for the request.
type NotificationSet struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#notificationSet".
	Kind string `json:"kind,omitempty"`

	// Notification: The notifications received, or empty if no
	// notifications are present.
	Notification []*Notification `json:"notification,omitempty"`

	// NotificationSetId: The notification set ID, required to mark the
	// notification as received with the Enterprises.AcknowledgeNotification
	// API. This will be omitted if no notifications are present.
	NotificationSetId string `json:"notificationSetId,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NotificationSet) MarshalJSON() ([]byte, error) {
	type noMethod NotificationSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type PageInfo struct {
	ResultPerPage int64 `json:"resultPerPage,omitempty"`

	StartIndex int64 `json:"startIndex,omitempty"`

	TotalResults int64 `json:"totalResults,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ResultPerPage") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ResultPerPage") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PageInfo) MarshalJSON() ([]byte, error) {
	type noMethod PageInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Permission: A Permissions resource represents some extra capability,
// to be granted to an Android app, which requires explicit consent. An
// enterprise admin must consent to these permissions on behalf of their
// users before an entitlement for the app can be created.
//
// The permissions collection is read-only. The information provided for
// each permission (localized name and description) is intended to be
// used in the MDM user interface when obtaining consent from the
// enterprise.
type Permission struct {
	// Description: A longer description of the Permissions resource, giving
	// more details of what it affects.
	Description string `json:"description,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#permission".
	Kind string `json:"kind,omitempty"`

	// Name: The name of the permission.
	Name string `json:"name,omitempty"`

	// PermissionId: An opaque string uniquely identifying the permission.
	PermissionId string `json:"permissionId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *Permission) MarshalJSON() ([]byte, error) {
	type noMethod Permission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Product: A Products resource represents an app in the Google Play
// store that is available to at least some users in the enterprise.
// (Some apps are restricted to a single enterprise, and no information
// about them is made available outside that enterprise.)
//
// The information provided for each product (localized name, icon, link
// to the full Google Play details page) is intended to allow a basic
// representation of the product within an EMM user interface.
type Product struct {
	// AppVersion: App versions currently available for this product.
	AppVersion []*AppVersion `json:"appVersion,omitempty"`

	// AuthorName: The name of the author of the product (for example, the
	// app developer).
	AuthorName string `json:"authorName,omitempty"`

	// DetailsUrl: A link to the (consumer) Google Play details page for the
	// product.
	DetailsUrl string `json:"detailsUrl,omitempty"`

	// DistributionChannel: How and to whom the package is made available.
	// The value publicGoogleHosted means that the package is available
	// through the Play store and not restricted to a specific enterprise.
	// The value privateGoogleHosted means that the package is a private app
	// (restricted to an enterprise) but hosted by Google. The value
	// privateSelfHosted means that the package is a private app (restricted
	// to an enterprise) and is privately hosted.
	DistributionChannel string `json:"distributionChannel,omitempty"`

	// IconUrl: A link to an image that can be used as an icon for the
	// product. This image is suitable for use at up to 512px x 512px.
	IconUrl string `json:"iconUrl,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#product".
	Kind string `json:"kind,omitempty"`

	// ProductId: A string of the form app:<package name>. For example,
	// app:com.google.android.gm represents the Gmail app.
	ProductId string `json:"productId,omitempty"`

	// ProductPricing: Whether this product is free, free with in-app
	// purchases, or paid. If the pricing is unknown, this means the product
	// is not generally available anymore (even though it might still be
	// available to people who own it).
	ProductPricing string `json:"productPricing,omitempty"`

	// RequiresContainerApp: Deprecated.
	RequiresContainerApp bool `json:"requiresContainerApp,omitempty"`

	// SigningCertificate: The certificate used to sign this product.
	SigningCertificate *ProductSigningCertificate `json:"signingCertificate,omitempty"`

	// SmallIconUrl: A link to a smaller image that can be used as an icon
	// for the product. This image is suitable for use at up to 128px x
	// 128px.
	SmallIconUrl string `json:"smallIconUrl,omitempty"`

	// Title: The name of the product.
	Title string `json:"title,omitempty"`

	// WorkDetailsUrl: A link to the managed Google Play details page for
	// the product, for use by an Enterprise admin.
	WorkDetailsUrl string `json:"workDetailsUrl,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AppVersion") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppVersion") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Product) MarshalJSON() ([]byte, error) {
	type noMethod Product
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductApprovalEvent: An event generated when a product's approval
// status is changed.
type ProductApprovalEvent struct {
	// Approved: Whether the product was approved or unapproved. This field
	// will always be present.
	Approved string `json:"approved,omitempty"`

	// ProductId: The id of the product (e.g. "app:com.google.android.gm")
	// for which the approval status has changed. This field will always be
	// present.
	ProductId string `json:"productId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Approved") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Approved") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductApprovalEvent) MarshalJSON() ([]byte, error) {
	type noMethod ProductApprovalEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductAvailabilityChangeEvent: An event generated whenever a
// product's availability changes.
type ProductAvailabilityChangeEvent struct {
	// AvailabilityStatus: The new state of the product. This field will
	// always be present.
	AvailabilityStatus string `json:"availabilityStatus,omitempty"`

	// ProductId: The id of the product (e.g. "app:com.google.android.gm")
	// for which the product availability changed. This field will always be
	// present.
	ProductId string `json:"productId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AvailabilityStatus")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AvailabilityStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ProductAvailabilityChangeEvent) MarshalJSON() ([]byte, error) {
	type noMethod ProductAvailabilityChangeEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductPermission: A product permissions resource represents the set
// of permissions required by a specific app and whether or not they
// have been accepted by an enterprise admin.
//
// The API can be used to read the set of permissions, and also to
// update the set to indicate that permissions have been accepted.
type ProductPermission struct {
	// PermissionId: An opaque string uniquely identifying the permission.
	PermissionId string `json:"permissionId,omitempty"`

	// State: Whether the permission has been accepted or not.
	State string `json:"state,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PermissionId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PermissionId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductPermission) MarshalJSON() ([]byte, error) {
	type noMethod ProductPermission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductPermissions: Information about the permissions required by a
// specific app and whether they have been accepted by the enterprise.
type ProductPermissions struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#productPermissions".
	Kind string `json:"kind,omitempty"`

	// Permission: The permissions required by the app.
	Permission []*ProductPermission `json:"permission,omitempty"`

	// ProductId: The ID of the app that the permissions relate to, e.g.
	// "app:com.google.android.gm".
	ProductId string `json:"productId,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductPermissions) MarshalJSON() ([]byte, error) {
	type noMethod ProductPermissions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductSet: A set of products.
type ProductSet struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#productSet".
	Kind string `json:"kind,omitempty"`

	// ProductId: The list of product IDs making up the set of products.
	ProductId []string `json:"productId,omitempty"`

	// ProductSetBehavior: The interpretation of this product set. "unknown"
	// should never be sent and is ignored if received. "whitelist" means
	// that the user is entitled to access the product set. "includeAll"
	// means that all products are accessible, including products that are
	// approved, products with revoked approval, and products that have
	// never been approved. "allApproved" means that the user is entitled to
	// access all products that are approved for the enterprise. If the
	// value is "allApproved" or "includeAll", the productId field is
	// ignored. If no value is provided, it is interpreted as "whitelist"
	// for backwards compatibility. Further "allApproved" or "includeAll"
	// does not enable automatic visibility of "alpha" or "beta" tracks for
	// Android app. Use ProductVisibility to enable "alpha" or "beta" tracks
	// per user.
	ProductSetBehavior string `json:"productSetBehavior,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductSet) MarshalJSON() ([]byte, error) {
	type noMethod ProductSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductSigningCertificate struct {
	// CertificateHashSha1: The base64 urlsafe encoded SHA1 hash of the
	// certificate. (This field is deprecated in favor of SHA2-256. It
	// should not be used and may be removed at any time.)
	CertificateHashSha1 string `json:"certificateHashSha1,omitempty"`

	// CertificateHashSha256: The base64 urlsafe encoded SHA2-256 hash of
	// the certificate.
	CertificateHashSha256 string `json:"certificateHashSha256,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CertificateHashSha1")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CertificateHashSha1") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ProductSigningCertificate) MarshalJSON() ([]byte, error) {
	type noMethod ProductSigningCertificate
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductsApproveRequest struct {
	// ApprovalUrlInfo: The approval URL that was shown to the user. Only
	// the permissions shown to the user with that URL will be accepted,
	// which may not be the product's entire set of permissions. For
	// example, the URL may only display new permissions from an update
	// after the product was approved, or not include new permissions if the
	// product was updated since the URL was generated.
	ApprovalUrlInfo *ApprovalUrlInfo `json:"approvalUrlInfo,omitempty"`

	// ApprovedPermissions: Sets how new permission requests for the product
	// are handled. "allPermissions" automatically approves all current and
	// future permissions for the product. "currentPermissionsOnly" approves
	// the current set of permissions for the product, but any future
	// permissions added through updates will require manual reapproval. If
	// not specified, only the current set of permissions will be approved.
	ApprovedPermissions string `json:"approvedPermissions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApprovalUrlInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApprovalUrlInfo") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ProductsApproveRequest) MarshalJSON() ([]byte, error) {
	type noMethod ProductsApproveRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ProductsGenerateApprovalUrlResponse struct {
	// Url: A URL that can be rendered in an iframe to display the
	// permissions (if any) of a product. This URL can be used to approve
	// the product only once and only within 24 hours of being generated,
	// using the Products.approve call. If the product is currently
	// unapproved and has no permissions, this URL will point to an empty
	// page. If the product is currently approved, a URL will only be
	// generated if that product has added permissions since it was last
	// approved, and the URL will only display those new permissions that
	// have not yet been accepted.
	Url string `json:"url,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Url") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Url") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductsGenerateApprovalUrlResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductsGenerateApprovalUrlResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProductsListResponse: The matching products.
type ProductsListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#productsListResponse".
	Kind string `json:"kind,omitempty"`

	// PageInfo: General pagination information.
	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// Product: Information about a product (e.g. an app) in the Google Play
	// store, for display to an enterprise admin.
	Product []*Product `json:"product,omitempty"`

	// TokenPagination: Pagination information for token pagination.
	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProductsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ServiceAccount: A service account identity, including the name and
// credentials that can be used to authenticate as the service account.
type ServiceAccount struct {
	// Key: Credentials that can be used to authenticate as this
	// ServiceAccount.
	Key *ServiceAccountKey `json:"key,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#serviceAccount".
	Kind string `json:"kind,omitempty"`

	// Name: The account name of the service account, in the form of an
	// email address. Assigned by the server.
	Name string `json:"name,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *ServiceAccount) MarshalJSON() ([]byte, error) {
	type noMethod ServiceAccount
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ServiceAccountKey: Credentials that can be used to authenticate as a
// service account.
type ServiceAccountKey struct {
	// Data: The body of the private key credentials file, in string format.
	// This is only populated when the ServiceAccountKey is created, and is
	// not stored by Google.
	Data string `json:"data,omitempty"`

	// Id: An opaque, unique identifier for this ServiceAccountKey. Assigned
	// by the server.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#serviceAccountKey".
	Kind string `json:"kind,omitempty"`

	// PublicData: Public key data for the credentials file. This is an
	// X.509 cert. If you are using the googleCredentials key type, this is
	// identical to the cert that can be retrieved by using the X.509 cert
	// url inside of the credentials file.
	PublicData string `json:"publicData,omitempty"`

	// Type: The file format of the generated key data.
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Data") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ServiceAccountKey) MarshalJSON() ([]byte, error) {
	type noMethod ServiceAccountKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ServiceAccountKeysListResponse struct {
	// ServiceAccountKey: The service account credentials.
	ServiceAccountKey []*ServiceAccountKey `json:"serviceAccountKey,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ServiceAccountKey")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ServiceAccountKey") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ServiceAccountKeysListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ServiceAccountKeysListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SignupInfo: A resource returned by the GenerateSignupUrl API, which
// contains the Signup URL and Completion Token.
type SignupInfo struct {
	// CompletionToken: An opaque token that will be required, along with
	// the Enterprise Token, for obtaining the enterprise resource from
	// CompleteSignup.
	CompletionToken string `json:"completionToken,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#signupInfo".
	Kind string `json:"kind,omitempty"`

	// Url: A URL under which the Admin can sign up for an enterprise. The
	// page pointed to cannot be rendered in an iframe.
	Url string `json:"url,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CompletionToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompletionToken") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SignupInfo) MarshalJSON() ([]byte, error) {
	type noMethod SignupInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// StoreCluster: Definition of a managed Google Play store cluster, a
// list of products displayed as part of a store page.
type StoreCluster struct {
	// Id: Unique ID of this cluster. Assigned by the server. Immutable once
	// assigned.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#storeCluster".
	Kind string `json:"kind,omitempty"`

	// Name: Ordered list of localized strings giving the name of this page.
	// The text displayed is the one that best matches the user locale, or
	// the first entry if there is no good match. There needs to be at least
	// one entry.
	Name []*LocalizedText `json:"name,omitempty"`

	// OrderInPage: String (US-ASCII only) used to determine order of this
	// cluster within the parent page's elements. Page elements are sorted
	// in lexicographic order of this field. Duplicated values are allowed,
	// but ordering between elements with duplicate order is undefined.
	//
	// The value of this field is never visible to a user, it is used solely
	// for the purpose of defining an ordering. Maximum length is 256
	// characters.
	OrderInPage string `json:"orderInPage,omitempty"`

	// ProductId: List of products in the order they are displayed in the
	// cluster. There should not be duplicates within a cluster.
	ProductId []string `json:"productId,omitempty"`

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

func (s *StoreCluster) MarshalJSON() ([]byte, error) {
	type noMethod StoreCluster
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// StoreLayout: General setting for the managed Google Play store
// layout, currently only specifying the page to display the first time
// the store is opened.
type StoreLayout struct {
	// HomepageId: The ID of the store page to be used as the homepage. The
	// homepage is the first page shown in the managed Google Play
	// Store.
	//
	// Not specifying a homepage is equivalent to setting the store layout
	// type to "basic".
	HomepageId string `json:"homepageId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#storeLayout".
	Kind string `json:"kind,omitempty"`

	// StoreLayoutType: The store layout type. By default, this value is set
	// to "basic" if the homepageId field is not set, and to "custom"
	// otherwise. If set to "basic", the layout will consist of all approved
	// apps that have been whitelisted for the user.
	StoreLayoutType string `json:"storeLayoutType,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "HomepageId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HomepageId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *StoreLayout) MarshalJSON() ([]byte, error) {
	type noMethod StoreLayout
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// StoreLayoutClustersListResponse: The store page resources for the
// enterprise.
type StoreLayoutClustersListResponse struct {
	// Cluster: A store cluster of an enterprise.
	Cluster []*StoreCluster `json:"cluster,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#storeLayoutClustersListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Cluster") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cluster") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *StoreLayoutClustersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod StoreLayoutClustersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// StoreLayoutPagesListResponse: The store page resources for the
// enterprise.
type StoreLayoutPagesListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#storeLayoutPagesListResponse".
	Kind string `json:"kind,omitempty"`

	// Page: A store page of an enterprise.
	Page []*StorePage `json:"page,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *StoreLayoutPagesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod StoreLayoutPagesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// StorePage: Definition of a managed Google Play store page, made of a
// localized name and links to other pages. A page also contains
// clusters defined as a subcollection.
type StorePage struct {
	// Id: Unique ID of this page. Assigned by the server. Immutable once
	// assigned.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#storePage".
	Kind string `json:"kind,omitempty"`

	// Link: Ordered list of pages a user should be able to reach from this
	// page. The pages must exist, must not be this page, and once a link is
	// created the page linked to cannot be deleted until all links to it
	// are removed. It is recommended that the basic pages are created
	// first, before adding the links between pages.
	//
	// No attempt is made to verify that all pages are reachable from the
	// homepage.
	Link []string `json:"link,omitempty"`

	// Name: Ordered list of localized strings giving the name of this page.
	// The text displayed is the one that best matches the user locale, or
	// the first entry if there is no good match. There needs to be at least
	// one entry.
	Name []*LocalizedText `json:"name,omitempty"`

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

func (s *StorePage) MarshalJSON() ([]byte, error) {
	type noMethod StorePage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TokenPagination struct {
	NextPageToken string `json:"nextPageToken,omitempty"`

	PreviousPageToken string `json:"previousPageToken,omitempty"`

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

func (s *TokenPagination) MarshalJSON() ([]byte, error) {
	type noMethod TokenPagination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// User: A Users resource represents an account associated with an
// enterprise. The account may be specific to a device or to an
// individual user (who can then use the account across multiple
// devices). The account may provide access to managed Google Play only,
// or to other Google services, depending on the identity model:
// - The Google managed domain identity model requires synchronization
// to Google account sources (via primaryEmail).
// - The managed Google Play Accounts identity model provides a dynamic
// means for enterprises to create user or device accounts as needed.
// These accounts provide access to managed Google Play.
type User struct {
	// AccountIdentifier: A unique identifier you create for this user, such
	// as "user342" or "asset#44418". Do not use personally identifiable
	// information (PII) for this property. Must always be set for
	// EMM-managed users. Not set for Google-managed users.
	AccountIdentifier string `json:"accountIdentifier,omitempty"`

	// AccountType: The type of account that this user represents. A
	// userAccount can be installed on multiple devices, but a deviceAccount
	// is specific to a single device. An EMM-managed user (emmManaged) can
	// be either type (userAccount, deviceAccount), but a Google-managed
	// user (googleManaged) is always a userAccount.
	AccountType string `json:"accountType,omitempty"`

	// DisplayName: The name that will appear in user interfaces. Setting
	// this property is optional when creating EMM-managed users. If you do
	// set this property, use something generic about the organization (such
	// as "Example, Inc.") or your name (as EMM). Not used for
	// Google-managed user accounts.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The unique ID for the user.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#user".
	Kind string `json:"kind,omitempty"`

	// ManagementType: The entity that manages the user. With googleManaged
	// users, the source of truth is Google so EMMs have to make sure a
	// Google Account exists for the user. With emmManaged users, the EMM is
	// in charge.
	ManagementType string `json:"managementType,omitempty"`

	// PrimaryEmail: The user's primary email address, for example,
	// "jsmith@example.com". Will always be set for Google managed users and
	// not set for EMM managed users.
	PrimaryEmail string `json:"primaryEmail,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountIdentifier")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountIdentifier") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *User) MarshalJSON() ([]byte, error) {
	type noMethod User
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UserToken: A UserToken is used by a user when setting up a managed
// device or profile with their managed Google Play account on a device.
// When the user enters their email address and token (activation code)
// the appropriate EMM app can be automatically downloaded.
type UserToken struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#userToken".
	Kind string `json:"kind,omitempty"`

	// Token: The token (activation code) to be entered by the user. This
	// consists of a sequence of decimal digits. Note that the leading digit
	// may be 0.
	Token string `json:"token,omitempty"`

	// UserId: The unique ID for the user.
	UserId string `json:"userId,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UserToken) MarshalJSON() ([]byte, error) {
	type noMethod UserToken
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UsersListResponse: The matching user resources.
type UsersListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#usersListResponse".
	Kind string `json:"kind,omitempty"`

	// User: A user of an enterprise.
	User []*User `json:"user,omitempty"`

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

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UsersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod UsersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "androidenterprise.devices.get":

type DevicesGetCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the details of a device.
func (r *DevicesService) Get(enterpriseId string, userId string, deviceId string) *DevicesGetCall {
	c := &DevicesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DevicesGetCall) Fields(s ...googleapi.Field) *DevicesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DevicesGetCall) IfNoneMatch(entityTag string) *DevicesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DevicesGetCall) Context(ctx context.Context) *DevicesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DevicesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DevicesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.devices.get" call.
// Exactly one of *Device or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Device.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DevicesGetCall) Do(opts ...googleapi.CallOption) (*Device, error) {
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
	ret := &Device{
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
	//   "description": "Retrieves the details of a device.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.devices.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}",
	//   "response": {
	//     "$ref": "Device"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.devices.getState":

type DevicesGetStateCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetState: Retrieves whether a device's access to Google services is
// enabled or disabled. The device state takes effect only if enforcing
// EMM policies on Android devices is enabled in the Google Admin
// Console. Otherwise, the device state is ignored and all devices are
// allowed access to Google services. This is only supported for
// Google-managed users.
func (r *DevicesService) GetState(enterpriseId string, userId string, deviceId string) *DevicesGetStateCall {
	c := &DevicesGetStateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DevicesGetStateCall) Fields(s ...googleapi.Field) *DevicesGetStateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DevicesGetStateCall) IfNoneMatch(entityTag string) *DevicesGetStateCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DevicesGetStateCall) Context(ctx context.Context) *DevicesGetStateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DevicesGetStateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DevicesGetStateCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/state")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.devices.getState" call.
// Exactly one of *DeviceState or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DeviceState.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DevicesGetStateCall) Do(opts ...googleapi.CallOption) (*DeviceState, error) {
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
	ret := &DeviceState{
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
	//   "description": "Retrieves whether a device's access to Google services is enabled or disabled. The device state takes effect only if enforcing EMM policies on Android devices is enabled in the Google Admin Console. Otherwise, the device state is ignored and all devices are allowed access to Google services. This is only supported for Google-managed users.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.devices.getState",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/state",
	//   "response": {
	//     "$ref": "DeviceState"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.devices.list":

type DevicesListCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the IDs of all of a user's devices.
func (r *DevicesService) List(enterpriseId string, userId string) *DevicesListCall {
	c := &DevicesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DevicesListCall) Fields(s ...googleapi.Field) *DevicesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DevicesListCall) IfNoneMatch(entityTag string) *DevicesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DevicesListCall) Context(ctx context.Context) *DevicesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DevicesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DevicesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.devices.list" call.
// Exactly one of *DevicesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DevicesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DevicesListCall) Do(opts ...googleapi.CallOption) (*DevicesListResponse, error) {
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
	ret := &DevicesListResponse{
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
	//   "description": "Retrieves the IDs of all of a user's devices.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.devices.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices",
	//   "response": {
	//     "$ref": "DevicesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.devices.setState":

type DevicesSetStateCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	devicestate  *DeviceState
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// SetState: Sets whether a device's access to Google services is
// enabled or disabled. The device state takes effect only if enforcing
// EMM policies on Android devices is enabled in the Google Admin
// Console. Otherwise, the device state is ignored and all devices are
// allowed access to Google services. This is only supported for
// Google-managed users.
func (r *DevicesService) SetState(enterpriseId string, userId string, deviceId string, devicestate *DeviceState) *DevicesSetStateCall {
	c := &DevicesSetStateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.devicestate = devicestate
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DevicesSetStateCall) Fields(s ...googleapi.Field) *DevicesSetStateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DevicesSetStateCall) Context(ctx context.Context) *DevicesSetStateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DevicesSetStateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DevicesSetStateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.devicestate)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/state")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.devices.setState" call.
// Exactly one of *DeviceState or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DeviceState.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DevicesSetStateCall) Do(opts ...googleapi.CallOption) (*DeviceState, error) {
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
	ret := &DeviceState{
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
	//   "description": "Sets whether a device's access to Google services is enabled or disabled. The device state takes effect only if enforcing EMM policies on Android devices is enabled in the Google Admin Console. Otherwise, the device state is ignored and all devices are allowed access to Google services. This is only supported for Google-managed users.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.devices.setState",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/state",
	//   "request": {
	//     "$ref": "DeviceState"
	//   },
	//   "response": {
	//     "$ref": "DeviceState"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.acknowledgeNotificationSet":

type EnterprisesAcknowledgeNotificationSetCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// AcknowledgeNotificationSet: Acknowledges notifications that were
// received from Enterprises.PullNotificationSet to prevent subsequent
// calls from returning the same notifications.
func (r *EnterprisesService) AcknowledgeNotificationSet() *EnterprisesAcknowledgeNotificationSetCall {
	c := &EnterprisesAcknowledgeNotificationSetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// NotificationSetId sets the optional parameter "notificationSetId":
// The notification set ID as returned by
// Enterprises.PullNotificationSet. This must be provided.
func (c *EnterprisesAcknowledgeNotificationSetCall) NotificationSetId(notificationSetId string) *EnterprisesAcknowledgeNotificationSetCall {
	c.urlParams_.Set("notificationSetId", notificationSetId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesAcknowledgeNotificationSetCall) Fields(s ...googleapi.Field) *EnterprisesAcknowledgeNotificationSetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesAcknowledgeNotificationSetCall) Context(ctx context.Context) *EnterprisesAcknowledgeNotificationSetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesAcknowledgeNotificationSetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesAcknowledgeNotificationSetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/acknowledgeNotificationSet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.acknowledgeNotificationSet" call.
func (c *EnterprisesAcknowledgeNotificationSetCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Acknowledges notifications that were received from Enterprises.PullNotificationSet to prevent subsequent calls from returning the same notifications.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.acknowledgeNotificationSet",
	//   "parameters": {
	//     "notificationSetId": {
	//       "description": "The notification set ID as returned by Enterprises.PullNotificationSet. This must be provided.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/acknowledgeNotificationSet",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.completeSignup":

type EnterprisesCompleteSignupCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// CompleteSignup: Completes the signup flow, by specifying the
// Completion token and Enterprise token. This request must not be
// called multiple times for a given Enterprise Token.
func (r *EnterprisesService) CompleteSignup() *EnterprisesCompleteSignupCall {
	c := &EnterprisesCompleteSignupCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// CompletionToken sets the optional parameter "completionToken": The
// Completion token initially returned by GenerateSignupUrl.
func (c *EnterprisesCompleteSignupCall) CompletionToken(completionToken string) *EnterprisesCompleteSignupCall {
	c.urlParams_.Set("completionToken", completionToken)
	return c
}

// EnterpriseToken sets the optional parameter "enterpriseToken": The
// Enterprise token appended to the Callback URL.
func (c *EnterprisesCompleteSignupCall) EnterpriseToken(enterpriseToken string) *EnterprisesCompleteSignupCall {
	c.urlParams_.Set("enterpriseToken", enterpriseToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesCompleteSignupCall) Fields(s ...googleapi.Field) *EnterprisesCompleteSignupCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesCompleteSignupCall) Context(ctx context.Context) *EnterprisesCompleteSignupCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesCompleteSignupCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesCompleteSignupCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/completeSignup")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.completeSignup" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesCompleteSignupCall) Do(opts ...googleapi.CallOption) (*Enterprise, error) {
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
	ret := &Enterprise{
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
	//   "description": "Completes the signup flow, by specifying the Completion token and Enterprise token. This request must not be called multiple times for a given Enterprise Token.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.completeSignup",
	//   "parameters": {
	//     "completionToken": {
	//       "description": "The Completion token initially returned by GenerateSignupUrl.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "enterpriseToken": {
	//       "description": "The Enterprise token appended to the Callback URL.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/completeSignup",
	//   "response": {
	//     "$ref": "Enterprise"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.createWebToken":

type EnterprisesCreateWebTokenCall struct {
	s                         *Service
	enterpriseId              string
	administratorwebtokenspec *AdministratorWebTokenSpec
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// CreateWebToken: Returns a unique token to access an embeddable UI. To
// generate a web UI, pass the generated token into the managed Google
// Play javascript API. Each token may only be used to start one UI
// session. See the javascript API documentation for further
// information.
func (r *EnterprisesService) CreateWebToken(enterpriseId string, administratorwebtokenspec *AdministratorWebTokenSpec) *EnterprisesCreateWebTokenCall {
	c := &EnterprisesCreateWebTokenCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.administratorwebtokenspec = administratorwebtokenspec
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesCreateWebTokenCall) Fields(s ...googleapi.Field) *EnterprisesCreateWebTokenCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesCreateWebTokenCall) Context(ctx context.Context) *EnterprisesCreateWebTokenCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesCreateWebTokenCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesCreateWebTokenCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.administratorwebtokenspec)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/createWebToken")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.createWebToken" call.
// Exactly one of *AdministratorWebToken or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AdministratorWebToken.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesCreateWebTokenCall) Do(opts ...googleapi.CallOption) (*AdministratorWebToken, error) {
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
	ret := &AdministratorWebToken{
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
	//   "description": "Returns a unique token to access an embeddable UI. To generate a web UI, pass the generated token into the managed Google Play javascript API. Each token may only be used to start one UI session. See the javascript API documentation for further information.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.createWebToken",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/createWebToken",
	//   "request": {
	//     "$ref": "AdministratorWebTokenSpec"
	//   },
	//   "response": {
	//     "$ref": "AdministratorWebToken"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.delete":

type EnterprisesDeleteCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deletes the binding between the EMM and enterprise. This is
// now deprecated. Use this method only to unenroll customers that were
// previously enrolled with the insert call, then enroll them again with
// the enroll call.
func (r *EnterprisesService) Delete(enterpriseId string) *EnterprisesDeleteCall {
	c := &EnterprisesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDeleteCall) Fields(s ...googleapi.Field) *EnterprisesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDeleteCall) Context(ctx context.Context) *EnterprisesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.delete" call.
func (c *EnterprisesDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the binding between the EMM and enterprise. This is now deprecated. Use this method only to unenroll customers that were previously enrolled with the insert call, then enroll them again with the enroll call.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.enterprises.delete",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.enroll":

type EnterprisesEnrollCall struct {
	s          *Service
	enterprise *Enterprise
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Enroll: Enrolls an enterprise with the calling EMM.
func (r *EnterprisesService) Enroll(token string, enterprise *Enterprise) *EnterprisesEnrollCall {
	c := &EnterprisesEnrollCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("token", token)
	c.enterprise = enterprise
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesEnrollCall) Fields(s ...googleapi.Field) *EnterprisesEnrollCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesEnrollCall) Context(ctx context.Context) *EnterprisesEnrollCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesEnrollCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesEnrollCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enterprise)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/enroll")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.enroll" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesEnrollCall) Do(opts ...googleapi.CallOption) (*Enterprise, error) {
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
	ret := &Enterprise{
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
	//   "description": "Enrolls an enterprise with the calling EMM.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.enroll",
	//   "parameterOrder": [
	//     "token"
	//   ],
	//   "parameters": {
	//     "token": {
	//       "description": "The token provided by the enterprise to register the EMM.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/enroll",
	//   "request": {
	//     "$ref": "Enterprise"
	//   },
	//   "response": {
	//     "$ref": "Enterprise"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.generateSignupUrl":

type EnterprisesGenerateSignupUrlCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// GenerateSignupUrl: Generates a sign-up URL.
func (r *EnterprisesService) GenerateSignupUrl() *EnterprisesGenerateSignupUrlCall {
	c := &EnterprisesGenerateSignupUrlCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// CallbackUrl sets the optional parameter "callbackUrl": The callback
// URL to which the Admin will be redirected after successfully creating
// an enterprise. Before redirecting there the system will add a single
// query parameter to this URL named "enterpriseToken" which will
// contain an opaque token to be used for the CompleteSignup
// request.
// Beware that this means that the URL will be parsed, the parameter
// added and then a new URL formatted, i.e. there may be some minor
// formatting changes and, more importantly, the URL must be well-formed
// so that it can be parsed.
func (c *EnterprisesGenerateSignupUrlCall) CallbackUrl(callbackUrl string) *EnterprisesGenerateSignupUrlCall {
	c.urlParams_.Set("callbackUrl", callbackUrl)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesGenerateSignupUrlCall) Fields(s ...googleapi.Field) *EnterprisesGenerateSignupUrlCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesGenerateSignupUrlCall) Context(ctx context.Context) *EnterprisesGenerateSignupUrlCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesGenerateSignupUrlCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesGenerateSignupUrlCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/signupUrl")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.generateSignupUrl" call.
// Exactly one of *SignupInfo or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *SignupInfo.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesGenerateSignupUrlCall) Do(opts ...googleapi.CallOption) (*SignupInfo, error) {
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
	ret := &SignupInfo{
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
	//   "description": "Generates a sign-up URL.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.generateSignupUrl",
	//   "parameters": {
	//     "callbackUrl": {
	//       "description": "The callback URL to which the Admin will be redirected after successfully creating an enterprise. Before redirecting there the system will add a single query parameter to this URL named \"enterpriseToken\" which will contain an opaque token to be used for the CompleteSignup request.\nBeware that this means that the URL will be parsed, the parameter added and then a new URL formatted, i.e. there may be some minor formatting changes and, more importantly, the URL must be well-formed so that it can be parsed.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/signupUrl",
	//   "response": {
	//     "$ref": "SignupInfo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.get":

type EnterprisesGetCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the name and domain of an enterprise.
func (r *EnterprisesService) Get(enterpriseId string) *EnterprisesGetCall {
	c := &EnterprisesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesGetCall) Fields(s ...googleapi.Field) *EnterprisesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesGetCall) IfNoneMatch(entityTag string) *EnterprisesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesGetCall) Context(ctx context.Context) *EnterprisesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.get" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesGetCall) Do(opts ...googleapi.CallOption) (*Enterprise, error) {
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
	ret := &Enterprise{
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
	//   "description": "Retrieves the name and domain of an enterprise.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.enterprises.get",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}",
	//   "response": {
	//     "$ref": "Enterprise"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.getAndroidDevicePolicyConfig":

type EnterprisesGetAndroidDevicePolicyConfigCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetAndroidDevicePolicyConfig: Returns the Android Device Policy
// config resource.
func (r *EnterprisesService) GetAndroidDevicePolicyConfig(enterpriseId string) *EnterprisesGetAndroidDevicePolicyConfigCall {
	c := &EnterprisesGetAndroidDevicePolicyConfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesGetAndroidDevicePolicyConfigCall) Fields(s ...googleapi.Field) *EnterprisesGetAndroidDevicePolicyConfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesGetAndroidDevicePolicyConfigCall) IfNoneMatch(entityTag string) *EnterprisesGetAndroidDevicePolicyConfigCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesGetAndroidDevicePolicyConfigCall) Context(ctx context.Context) *EnterprisesGetAndroidDevicePolicyConfigCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesGetAndroidDevicePolicyConfigCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesGetAndroidDevicePolicyConfigCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/androidDevicePolicyConfig")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.getAndroidDevicePolicyConfig" call.
// Exactly one of *AndroidDevicePolicyConfig or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AndroidDevicePolicyConfig.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesGetAndroidDevicePolicyConfigCall) Do(opts ...googleapi.CallOption) (*AndroidDevicePolicyConfig, error) {
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
	ret := &AndroidDevicePolicyConfig{
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
	//   "description": "Returns the Android Device Policy config resource.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.enterprises.getAndroidDevicePolicyConfig",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/androidDevicePolicyConfig",
	//   "response": {
	//     "$ref": "AndroidDevicePolicyConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.getServiceAccount":

type EnterprisesGetServiceAccountCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetServiceAccount: Returns a service account and credentials. The
// service account can be bound to the enterprise by calling setAccount.
// The service account is unique to this enterprise and EMM, and will be
// deleted if the enterprise is unbound. The credentials contain private
// key data and are not stored server-side.
//
// This method can only be called after calling Enterprises.Enroll or
// Enterprises.CompleteSignup, and before Enterprises.SetAccount; at
// other times it will return an error.
//
// Subsequent calls after the first will generate a new, unique set of
// credentials, and invalidate the previously generated
// credentials.
//
// Once the service account is bound to the enterprise, it can be
// managed using the serviceAccountKeys resource.
func (r *EnterprisesService) GetServiceAccount(enterpriseId string) *EnterprisesGetServiceAccountCall {
	c := &EnterprisesGetServiceAccountCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// KeyType sets the optional parameter "keyType": The type of credential
// to return with the service account. Required.
//
// Possible values:
//   "googleCredentials"
//   "pkcs12"
func (c *EnterprisesGetServiceAccountCall) KeyType(keyType string) *EnterprisesGetServiceAccountCall {
	c.urlParams_.Set("keyType", keyType)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesGetServiceAccountCall) Fields(s ...googleapi.Field) *EnterprisesGetServiceAccountCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesGetServiceAccountCall) IfNoneMatch(entityTag string) *EnterprisesGetServiceAccountCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesGetServiceAccountCall) Context(ctx context.Context) *EnterprisesGetServiceAccountCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesGetServiceAccountCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesGetServiceAccountCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/serviceAccount")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.getServiceAccount" call.
// Exactly one of *ServiceAccount or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ServiceAccount.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesGetServiceAccountCall) Do(opts ...googleapi.CallOption) (*ServiceAccount, error) {
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
	ret := &ServiceAccount{
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
	//   "description": "Returns a service account and credentials. The service account can be bound to the enterprise by calling setAccount. The service account is unique to this enterprise and EMM, and will be deleted if the enterprise is unbound. The credentials contain private key data and are not stored server-side.\n\nThis method can only be called after calling Enterprises.Enroll or Enterprises.CompleteSignup, and before Enterprises.SetAccount; at other times it will return an error.\n\nSubsequent calls after the first will generate a new, unique set of credentials, and invalidate the previously generated credentials.\n\nOnce the service account is bound to the enterprise, it can be managed using the serviceAccountKeys resource.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.enterprises.getServiceAccount",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "keyType": {
	//       "description": "The type of credential to return with the service account. Required.",
	//       "enum": [
	//         "googleCredentials",
	//         "pkcs12"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/serviceAccount",
	//   "response": {
	//     "$ref": "ServiceAccount"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.getStoreLayout":

type EnterprisesGetStoreLayoutCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetStoreLayout: Returns the store layout for the enterprise. If the
// store layout has not been set, returns "basic" as the store layout
// type and no homepage.
func (r *EnterprisesService) GetStoreLayout(enterpriseId string) *EnterprisesGetStoreLayoutCall {
	c := &EnterprisesGetStoreLayoutCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesGetStoreLayoutCall) Fields(s ...googleapi.Field) *EnterprisesGetStoreLayoutCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesGetStoreLayoutCall) IfNoneMatch(entityTag string) *EnterprisesGetStoreLayoutCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesGetStoreLayoutCall) Context(ctx context.Context) *EnterprisesGetStoreLayoutCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesGetStoreLayoutCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesGetStoreLayoutCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.getStoreLayout" call.
// Exactly one of *StoreLayout or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreLayout.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesGetStoreLayoutCall) Do(opts ...googleapi.CallOption) (*StoreLayout, error) {
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
	ret := &StoreLayout{
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
	//   "description": "Returns the store layout for the enterprise. If the store layout has not been set, returns \"basic\" as the store layout type and no homepage.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.enterprises.getStoreLayout",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout",
	//   "response": {
	//     "$ref": "StoreLayout"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.insert":

type EnterprisesInsertCall struct {
	s          *Service
	enterprise *Enterprise
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Establishes the binding between the EMM and an enterprise.
// This is now deprecated; use enroll instead.
func (r *EnterprisesService) Insert(token string, enterprise *Enterprise) *EnterprisesInsertCall {
	c := &EnterprisesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("token", token)
	c.enterprise = enterprise
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesInsertCall) Fields(s ...googleapi.Field) *EnterprisesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesInsertCall) Context(ctx context.Context) *EnterprisesInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enterprise)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.insert" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesInsertCall) Do(opts ...googleapi.CallOption) (*Enterprise, error) {
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
	ret := &Enterprise{
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
	//   "description": "Establishes the binding between the EMM and an enterprise. This is now deprecated; use enroll instead.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.insert",
	//   "parameterOrder": [
	//     "token"
	//   ],
	//   "parameters": {
	//     "token": {
	//       "description": "The token provided by the enterprise to register the EMM.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises",
	//   "request": {
	//     "$ref": "Enterprise"
	//   },
	//   "response": {
	//     "$ref": "Enterprise"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.list":

type EnterprisesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Looks up an enterprise by domain name. This is only supported
// for enterprises created via the Google-initiated creation flow.
// Lookup of the id is not needed for enterprises created via the
// EMM-initiated flow since the EMM learns the enterprise ID in the
// callback specified in the Enterprises.generateSignupUrl call.
func (r *EnterprisesService) List(domain string) *EnterprisesListCall {
	c := &EnterprisesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("domain", domain)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesListCall) Fields(s ...googleapi.Field) *EnterprisesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesListCall) IfNoneMatch(entityTag string) *EnterprisesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesListCall) Context(ctx context.Context) *EnterprisesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.list" call.
// Exactly one of *EnterprisesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *EnterprisesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesListCall) Do(opts ...googleapi.CallOption) (*EnterprisesListResponse, error) {
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
	ret := &EnterprisesListResponse{
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
	//   "description": "Looks up an enterprise by domain name. This is only supported for enterprises created via the Google-initiated creation flow. Lookup of the id is not needed for enterprises created via the EMM-initiated flow since the EMM learns the enterprise ID in the callback specified in the Enterprises.generateSignupUrl call.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.enterprises.list",
	//   "parameterOrder": [
	//     "domain"
	//   ],
	//   "parameters": {
	//     "domain": {
	//       "description": "The exact primary domain name of the enterprise to look up.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises",
	//   "response": {
	//     "$ref": "EnterprisesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.pullNotificationSet":

type EnterprisesPullNotificationSetCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// PullNotificationSet: Pulls and returns a notification set for the
// enterprises associated with the service account authenticated for the
// request. The notification set may be empty if no notification are
// pending.
// A notification set returned needs to be acknowledged within 20
// seconds by calling Enterprises.AcknowledgeNotificationSet, unless the
// notification set is empty.
// Notifications that are not acknowledged within the 20 seconds will
// eventually be included again in the response to another
// PullNotificationSet request, and those that are never acknowledged
// will ultimately be deleted according to the Google Cloud Platform
// Pub/Sub system policy.
// Multiple requests might be performed concurrently to retrieve
// notifications, in which case the pending notifications (if any) will
// be split among each caller, if any are pending.
// If no notifications are present, an empty notification list is
// returned. Subsequent requests may return more notifications once they
// become available.
func (r *EnterprisesService) PullNotificationSet() *EnterprisesPullNotificationSetCall {
	c := &EnterprisesPullNotificationSetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// RequestMode sets the optional parameter "requestMode": The request
// mode for pulling notifications.
// Specifying waitForNotifications will cause the request to block and
// wait until one or more notifications are present, or return an empty
// notification list if no notifications are present after some
// time.
// Speciying returnImmediately will cause the request to immediately
// return the pending notifications, or an empty list if no
// notifications are present.
// If omitted, defaults to waitForNotifications.
//
// Possible values:
//   "returnImmediately"
//   "waitForNotifications"
func (c *EnterprisesPullNotificationSetCall) RequestMode(requestMode string) *EnterprisesPullNotificationSetCall {
	c.urlParams_.Set("requestMode", requestMode)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesPullNotificationSetCall) Fields(s ...googleapi.Field) *EnterprisesPullNotificationSetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesPullNotificationSetCall) Context(ctx context.Context) *EnterprisesPullNotificationSetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesPullNotificationSetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesPullNotificationSetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/pullNotificationSet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.pullNotificationSet" call.
// Exactly one of *NotificationSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *NotificationSet.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesPullNotificationSetCall) Do(opts ...googleapi.CallOption) (*NotificationSet, error) {
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
	ret := &NotificationSet{
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
	//   "description": "Pulls and returns a notification set for the enterprises associated with the service account authenticated for the request. The notification set may be empty if no notification are pending.\nA notification set returned needs to be acknowledged within 20 seconds by calling Enterprises.AcknowledgeNotificationSet, unless the notification set is empty.\nNotifications that are not acknowledged within the 20 seconds will eventually be included again in the response to another PullNotificationSet request, and those that are never acknowledged will ultimately be deleted according to the Google Cloud Platform Pub/Sub system policy.\nMultiple requests might be performed concurrently to retrieve notifications, in which case the pending notifications (if any) will be split among each caller, if any are pending.\nIf no notifications are present, an empty notification list is returned. Subsequent requests may return more notifications once they become available.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.pullNotificationSet",
	//   "parameters": {
	//     "requestMode": {
	//       "description": "The request mode for pulling notifications.\nSpecifying waitForNotifications will cause the request to block and wait until one or more notifications are present, or return an empty notification list if no notifications are present after some time.\nSpeciying returnImmediately will cause the request to immediately return the pending notifications, or an empty list if no notifications are present.\nIf omitted, defaults to waitForNotifications.",
	//       "enum": [
	//         "returnImmediately",
	//         "waitForNotifications"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/pullNotificationSet",
	//   "response": {
	//     "$ref": "NotificationSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.sendTestPushNotification":

type EnterprisesSendTestPushNotificationCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// SendTestPushNotification: Sends a test notification to validate the
// EMM integration with the Google Cloud Pub/Sub service for this
// enterprise.
func (r *EnterprisesService) SendTestPushNotification(enterpriseId string) *EnterprisesSendTestPushNotificationCall {
	c := &EnterprisesSendTestPushNotificationCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesSendTestPushNotificationCall) Fields(s ...googleapi.Field) *EnterprisesSendTestPushNotificationCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesSendTestPushNotificationCall) Context(ctx context.Context) *EnterprisesSendTestPushNotificationCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesSendTestPushNotificationCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesSendTestPushNotificationCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/sendTestPushNotification")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.sendTestPushNotification" call.
// Exactly one of *EnterprisesSendTestPushNotificationResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *EnterprisesSendTestPushNotificationResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *EnterprisesSendTestPushNotificationCall) Do(opts ...googleapi.CallOption) (*EnterprisesSendTestPushNotificationResponse, error) {
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
	ret := &EnterprisesSendTestPushNotificationResponse{
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
	//   "description": "Sends a test notification to validate the EMM integration with the Google Cloud Pub/Sub service for this enterprise.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.sendTestPushNotification",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/sendTestPushNotification",
	//   "response": {
	//     "$ref": "EnterprisesSendTestPushNotificationResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.setAccount":

type EnterprisesSetAccountCall struct {
	s                 *Service
	enterpriseId      string
	enterpriseaccount *EnterpriseAccount
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// SetAccount: Sets the account that will be used to authenticate to the
// API as the enterprise.
func (r *EnterprisesService) SetAccount(enterpriseId string, enterpriseaccount *EnterpriseAccount) *EnterprisesSetAccountCall {
	c := &EnterprisesSetAccountCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.enterpriseaccount = enterpriseaccount
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesSetAccountCall) Fields(s ...googleapi.Field) *EnterprisesSetAccountCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesSetAccountCall) Context(ctx context.Context) *EnterprisesSetAccountCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesSetAccountCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesSetAccountCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enterpriseaccount)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/account")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.setAccount" call.
// Exactly one of *EnterpriseAccount or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *EnterpriseAccount.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesSetAccountCall) Do(opts ...googleapi.CallOption) (*EnterpriseAccount, error) {
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
	ret := &EnterpriseAccount{
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
	//   "description": "Sets the account that will be used to authenticate to the API as the enterprise.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.enterprises.setAccount",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/account",
	//   "request": {
	//     "$ref": "EnterpriseAccount"
	//   },
	//   "response": {
	//     "$ref": "EnterpriseAccount"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.setAndroidDevicePolicyConfig":

type EnterprisesSetAndroidDevicePolicyConfigCall struct {
	s                         *Service
	enterpriseId              string
	androiddevicepolicyconfig *AndroidDevicePolicyConfig
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// SetAndroidDevicePolicyConfig: Sets the Android Device Policy config
// resource. EMM may use this method to enable or disable Android Device
// Policy support for the specified enterprise. To learn more about
// managing devices and apps with Android Device Policy, see the Android
// Management API.
func (r *EnterprisesService) SetAndroidDevicePolicyConfig(enterpriseId string, androiddevicepolicyconfig *AndroidDevicePolicyConfig) *EnterprisesSetAndroidDevicePolicyConfigCall {
	c := &EnterprisesSetAndroidDevicePolicyConfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.androiddevicepolicyconfig = androiddevicepolicyconfig
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesSetAndroidDevicePolicyConfigCall) Fields(s ...googleapi.Field) *EnterprisesSetAndroidDevicePolicyConfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesSetAndroidDevicePolicyConfigCall) Context(ctx context.Context) *EnterprisesSetAndroidDevicePolicyConfigCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesSetAndroidDevicePolicyConfigCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesSetAndroidDevicePolicyConfigCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.androiddevicepolicyconfig)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/androidDevicePolicyConfig")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.setAndroidDevicePolicyConfig" call.
// Exactly one of *AndroidDevicePolicyConfig or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *AndroidDevicePolicyConfig.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesSetAndroidDevicePolicyConfigCall) Do(opts ...googleapi.CallOption) (*AndroidDevicePolicyConfig, error) {
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
	ret := &AndroidDevicePolicyConfig{
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
	//   "description": "Sets the Android Device Policy config resource. EMM may use this method to enable or disable Android Device Policy support for the specified enterprise. To learn more about managing devices and apps with Android Device Policy, see the Android Management API.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.enterprises.setAndroidDevicePolicyConfig",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/androidDevicePolicyConfig",
	//   "request": {
	//     "$ref": "AndroidDevicePolicyConfig"
	//   },
	//   "response": {
	//     "$ref": "AndroidDevicePolicyConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.setStoreLayout":

type EnterprisesSetStoreLayoutCall struct {
	s            *Service
	enterpriseId string
	storelayout  *StoreLayout
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// SetStoreLayout: Sets the store layout for the enterprise. By default,
// storeLayoutType is set to "basic" and the basic store layout is
// enabled. The basic layout only contains apps approved by the admin,
// and that have been added to the available product set for a user
// (using the  setAvailableProductSet call). Apps on the page are sorted
// in order of their product ID value. If you create a custom store
// layout (by setting storeLayoutType = "custom" and setting a
// homepage), the basic store layout is disabled.
func (r *EnterprisesService) SetStoreLayout(enterpriseId string, storelayout *StoreLayout) *EnterprisesSetStoreLayoutCall {
	c := &EnterprisesSetStoreLayoutCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.storelayout = storelayout
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesSetStoreLayoutCall) Fields(s ...googleapi.Field) *EnterprisesSetStoreLayoutCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesSetStoreLayoutCall) Context(ctx context.Context) *EnterprisesSetStoreLayoutCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesSetStoreLayoutCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesSetStoreLayoutCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storelayout)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.setStoreLayout" call.
// Exactly one of *StoreLayout or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreLayout.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesSetStoreLayoutCall) Do(opts ...googleapi.CallOption) (*StoreLayout, error) {
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
	ret := &StoreLayout{
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
	//   "description": "Sets the store layout for the enterprise. By default, storeLayoutType is set to \"basic\" and the basic store layout is enabled. The basic layout only contains apps approved by the admin, and that have been added to the available product set for a user (using the  setAvailableProductSet call). Apps on the page are sorted in order of their product ID value. If you create a custom store layout (by setting storeLayoutType = \"custom\" and setting a homepage), the basic store layout is disabled.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.enterprises.setStoreLayout",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout",
	//   "request": {
	//     "$ref": "StoreLayout"
	//   },
	//   "response": {
	//     "$ref": "StoreLayout"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.enterprises.unenroll":

type EnterprisesUnenrollCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Unenroll: Unenrolls an enterprise from the calling EMM.
func (r *EnterprisesService) Unenroll(enterpriseId string) *EnterprisesUnenrollCall {
	c := &EnterprisesUnenrollCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesUnenrollCall) Fields(s ...googleapi.Field) *EnterprisesUnenrollCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesUnenrollCall) Context(ctx context.Context) *EnterprisesUnenrollCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesUnenrollCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesUnenrollCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/unenroll")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.enterprises.unenroll" call.
func (c *EnterprisesUnenrollCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Unenrolls an enterprise from the calling EMM.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.unenroll",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/unenroll",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.entitlements.delete":

type EntitlementsDeleteCall struct {
	s             *Service
	enterpriseId  string
	userId        string
	entitlementId string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Delete: Removes an entitlement to an app for a user.
func (r *EntitlementsService) Delete(enterpriseId string, userId string, entitlementId string) *EntitlementsDeleteCall {
	c := &EntitlementsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.entitlementId = entitlementId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntitlementsDeleteCall) Fields(s ...googleapi.Field) *EntitlementsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntitlementsDeleteCall) Context(ctx context.Context) *EntitlementsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EntitlementsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EntitlementsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.entitlements.delete" call.
func (c *EntitlementsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Removes an entitlement to an app for a user.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.entitlements.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "entitlementId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entitlementId": {
	//       "description": "The ID of the entitlement (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.entitlements.get":

type EntitlementsGetCall struct {
	s             *Service
	enterpriseId  string
	userId        string
	entitlementId string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
	header_       http.Header
}

// Get: Retrieves details of an entitlement.
func (r *EntitlementsService) Get(enterpriseId string, userId string, entitlementId string) *EntitlementsGetCall {
	c := &EntitlementsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.entitlementId = entitlementId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntitlementsGetCall) Fields(s ...googleapi.Field) *EntitlementsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EntitlementsGetCall) IfNoneMatch(entityTag string) *EntitlementsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntitlementsGetCall) Context(ctx context.Context) *EntitlementsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EntitlementsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EntitlementsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.entitlements.get" call.
// Exactly one of *Entitlement or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Entitlement.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EntitlementsGetCall) Do(opts ...googleapi.CallOption) (*Entitlement, error) {
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
	ret := &Entitlement{
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
	//   "description": "Retrieves details of an entitlement.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.entitlements.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "entitlementId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entitlementId": {
	//       "description": "The ID of the entitlement (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}",
	//   "response": {
	//     "$ref": "Entitlement"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.entitlements.list":

type EntitlementsListCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all entitlements for the specified user. Only the ID is
// set.
func (r *EntitlementsService) List(enterpriseId string, userId string) *EntitlementsListCall {
	c := &EntitlementsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntitlementsListCall) Fields(s ...googleapi.Field) *EntitlementsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EntitlementsListCall) IfNoneMatch(entityTag string) *EntitlementsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntitlementsListCall) Context(ctx context.Context) *EntitlementsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EntitlementsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EntitlementsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.entitlements.list" call.
// Exactly one of *EntitlementsListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *EntitlementsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EntitlementsListCall) Do(opts ...googleapi.CallOption) (*EntitlementsListResponse, error) {
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
	ret := &EntitlementsListResponse{
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
	//   "description": "Lists all entitlements for the specified user. Only the ID is set.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.entitlements.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/entitlements",
	//   "response": {
	//     "$ref": "EntitlementsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.entitlements.patch":

type EntitlementsPatchCall struct {
	s             *Service
	enterpriseId  string
	userId        string
	entitlementId string
	entitlement   *Entitlement
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Patch: Adds or updates an entitlement to an app for a user. This
// method supports patch semantics.
func (r *EntitlementsService) Patch(enterpriseId string, userId string, entitlementId string, entitlement *Entitlement) *EntitlementsPatchCall {
	c := &EntitlementsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.entitlementId = entitlementId
	c.entitlement = entitlement
	return c
}

// Install sets the optional parameter "install": Set to true to also
// install the product on all the user's devices where possible. Failure
// to install on one or more devices will not prevent this operation
// from returning successfully, as long as the entitlement was
// successfully assigned to the user.
func (c *EntitlementsPatchCall) Install(install bool) *EntitlementsPatchCall {
	c.urlParams_.Set("install", fmt.Sprint(install))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntitlementsPatchCall) Fields(s ...googleapi.Field) *EntitlementsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntitlementsPatchCall) Context(ctx context.Context) *EntitlementsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EntitlementsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EntitlementsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entitlement)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.entitlements.patch" call.
// Exactly one of *Entitlement or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Entitlement.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EntitlementsPatchCall) Do(opts ...googleapi.CallOption) (*Entitlement, error) {
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
	ret := &Entitlement{
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
	//   "description": "Adds or updates an entitlement to an app for a user. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.entitlements.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "entitlementId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entitlementId": {
	//       "description": "The ID of the entitlement (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "install": {
	//       "description": "Set to true to also install the product on all the user's devices where possible. Failure to install on one or more devices will not prevent this operation from returning successfully, as long as the entitlement was successfully assigned to the user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}",
	//   "request": {
	//     "$ref": "Entitlement"
	//   },
	//   "response": {
	//     "$ref": "Entitlement"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.entitlements.update":

type EntitlementsUpdateCall struct {
	s             *Service
	enterpriseId  string
	userId        string
	entitlementId string
	entitlement   *Entitlement
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Update: Adds or updates an entitlement to an app for a user.
func (r *EntitlementsService) Update(enterpriseId string, userId string, entitlementId string, entitlement *Entitlement) *EntitlementsUpdateCall {
	c := &EntitlementsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.entitlementId = entitlementId
	c.entitlement = entitlement
	return c
}

// Install sets the optional parameter "install": Set to true to also
// install the product on all the user's devices where possible. Failure
// to install on one or more devices will not prevent this operation
// from returning successfully, as long as the entitlement was
// successfully assigned to the user.
func (c *EntitlementsUpdateCall) Install(install bool) *EntitlementsUpdateCall {
	c.urlParams_.Set("install", fmt.Sprint(install))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntitlementsUpdateCall) Fields(s ...googleapi.Field) *EntitlementsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntitlementsUpdateCall) Context(ctx context.Context) *EntitlementsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EntitlementsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EntitlementsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entitlement)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.entitlements.update" call.
// Exactly one of *Entitlement or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Entitlement.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EntitlementsUpdateCall) Do(opts ...googleapi.CallOption) (*Entitlement, error) {
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
	ret := &Entitlement{
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
	//   "description": "Adds or updates an entitlement to an app for a user.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.entitlements.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "entitlementId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "entitlementId": {
	//       "description": "The ID of the entitlement (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "install": {
	//       "description": "Set to true to also install the product on all the user's devices where possible. Failure to install on one or more devices will not prevent this operation from returning successfully, as long as the entitlement was successfully assigned to the user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}",
	//   "request": {
	//     "$ref": "Entitlement"
	//   },
	//   "response": {
	//     "$ref": "Entitlement"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.grouplicenses.get":

type GrouplicensesGetCall struct {
	s              *Service
	enterpriseId   string
	groupLicenseId string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
	header_        http.Header
}

// Get: Retrieves details of an enterprise's group license for a
// product.
func (r *GrouplicensesService) Get(enterpriseId string, groupLicenseId string) *GrouplicensesGetCall {
	c := &GrouplicensesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.groupLicenseId = groupLicenseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GrouplicensesGetCall) Fields(s ...googleapi.Field) *GrouplicensesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GrouplicensesGetCall) IfNoneMatch(entityTag string) *GrouplicensesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GrouplicensesGetCall) Context(ctx context.Context) *GrouplicensesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GrouplicensesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GrouplicensesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/groupLicenses/{groupLicenseId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":   c.enterpriseId,
		"groupLicenseId": c.groupLicenseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.grouplicenses.get" call.
// Exactly one of *GroupLicense or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *GroupLicense.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GrouplicensesGetCall) Do(opts ...googleapi.CallOption) (*GroupLicense, error) {
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
	ret := &GroupLicense{
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
	//   "description": "Retrieves details of an enterprise's group license for a product.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.grouplicenses.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "groupLicenseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "groupLicenseId": {
	//       "description": "The ID of the product the group license is for, e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/groupLicenses/{groupLicenseId}",
	//   "response": {
	//     "$ref": "GroupLicense"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.grouplicenses.list":

type GrouplicensesListCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves IDs of all products for which the enterprise has a
// group license.
func (r *GrouplicensesService) List(enterpriseId string) *GrouplicensesListCall {
	c := &GrouplicensesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GrouplicensesListCall) Fields(s ...googleapi.Field) *GrouplicensesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GrouplicensesListCall) IfNoneMatch(entityTag string) *GrouplicensesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GrouplicensesListCall) Context(ctx context.Context) *GrouplicensesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GrouplicensesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GrouplicensesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/groupLicenses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.grouplicenses.list" call.
// Exactly one of *GroupLicensesListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GroupLicensesListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GrouplicensesListCall) Do(opts ...googleapi.CallOption) (*GroupLicensesListResponse, error) {
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
	ret := &GroupLicensesListResponse{
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
	//   "description": "Retrieves IDs of all products for which the enterprise has a group license.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.grouplicenses.list",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/groupLicenses",
	//   "response": {
	//     "$ref": "GroupLicensesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.grouplicenseusers.list":

type GrouplicenseusersListCall struct {
	s              *Service
	enterpriseId   string
	groupLicenseId string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
	header_        http.Header
}

// List: Retrieves the IDs of the users who have been granted
// entitlements under the license.
func (r *GrouplicenseusersService) List(enterpriseId string, groupLicenseId string) *GrouplicenseusersListCall {
	c := &GrouplicenseusersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.groupLicenseId = groupLicenseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GrouplicenseusersListCall) Fields(s ...googleapi.Field) *GrouplicenseusersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GrouplicenseusersListCall) IfNoneMatch(entityTag string) *GrouplicenseusersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GrouplicenseusersListCall) Context(ctx context.Context) *GrouplicenseusersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GrouplicenseusersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GrouplicenseusersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/groupLicenses/{groupLicenseId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":   c.enterpriseId,
		"groupLicenseId": c.groupLicenseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.grouplicenseusers.list" call.
// Exactly one of *GroupLicenseUsersListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *GroupLicenseUsersListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GrouplicenseusersListCall) Do(opts ...googleapi.CallOption) (*GroupLicenseUsersListResponse, error) {
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
	ret := &GroupLicenseUsersListResponse{
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
	//   "description": "Retrieves the IDs of the users who have been granted entitlements under the license.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.grouplicenseusers.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "groupLicenseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "groupLicenseId": {
	//       "description": "The ID of the product the group license is for, e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/groupLicenses/{groupLicenseId}/users",
	//   "response": {
	//     "$ref": "GroupLicenseUsersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.installs.delete":

type InstallsDeleteCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	installId    string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Requests to remove an app from a device. A call to get or
// list will still show the app as installed on the device until it is
// actually removed.
func (r *InstallsService) Delete(enterpriseId string, userId string, deviceId string, installId string) *InstallsDeleteCall {
	c := &InstallsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.installId = installId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstallsDeleteCall) Fields(s ...googleapi.Field) *InstallsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstallsDeleteCall) Context(ctx context.Context) *InstallsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstallsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstallsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.installs.delete" call.
func (c *InstallsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Requests to remove an app from a device. A call to get or list will still show the app as installed on the device until it is actually removed.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.installs.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "installId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "installId": {
	//       "description": "The ID of the product represented by the install, e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.installs.get":

type InstallsGetCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	installId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves details of an installation of an app on a device.
func (r *InstallsService) Get(enterpriseId string, userId string, deviceId string, installId string) *InstallsGetCall {
	c := &InstallsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.installId = installId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstallsGetCall) Fields(s ...googleapi.Field) *InstallsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InstallsGetCall) IfNoneMatch(entityTag string) *InstallsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstallsGetCall) Context(ctx context.Context) *InstallsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstallsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstallsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.installs.get" call.
// Exactly one of *Install or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Install.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InstallsGetCall) Do(opts ...googleapi.CallOption) (*Install, error) {
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
	ret := &Install{
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
	//   "description": "Retrieves details of an installation of an app on a device.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.installs.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "installId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "installId": {
	//       "description": "The ID of the product represented by the install, e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}",
	//   "response": {
	//     "$ref": "Install"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.installs.list":

type InstallsListCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the details of all apps installed on the specified
// device.
func (r *InstallsService) List(enterpriseId string, userId string, deviceId string) *InstallsListCall {
	c := &InstallsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstallsListCall) Fields(s ...googleapi.Field) *InstallsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InstallsListCall) IfNoneMatch(entityTag string) *InstallsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstallsListCall) Context(ctx context.Context) *InstallsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstallsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstallsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.installs.list" call.
// Exactly one of *InstallsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *InstallsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InstallsListCall) Do(opts ...googleapi.CallOption) (*InstallsListResponse, error) {
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
	ret := &InstallsListResponse{
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
	//   "description": "Retrieves the details of all apps installed on the specified device.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.installs.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs",
	//   "response": {
	//     "$ref": "InstallsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.installs.patch":

type InstallsPatchCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	installId    string
	install      *Install
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Patch: Requests to install the latest version of an app to a device.
// If the app is already installed, then it is updated to the latest
// version if necessary. This method supports patch semantics.
func (r *InstallsService) Patch(enterpriseId string, userId string, deviceId string, installId string, install *Install) *InstallsPatchCall {
	c := &InstallsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.installId = installId
	c.install = install
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstallsPatchCall) Fields(s ...googleapi.Field) *InstallsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstallsPatchCall) Context(ctx context.Context) *InstallsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstallsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstallsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.install)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.installs.patch" call.
// Exactly one of *Install or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Install.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InstallsPatchCall) Do(opts ...googleapi.CallOption) (*Install, error) {
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
	ret := &Install{
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
	//   "description": "Requests to install the latest version of an app to a device. If the app is already installed, then it is updated to the latest version if necessary. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.installs.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "installId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "installId": {
	//       "description": "The ID of the product represented by the install, e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}",
	//   "request": {
	//     "$ref": "Install"
	//   },
	//   "response": {
	//     "$ref": "Install"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.installs.update":

type InstallsUpdateCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	installId    string
	install      *Install
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Requests to install the latest version of an app to a device.
// If the app is already installed, then it is updated to the latest
// version if necessary.
func (r *InstallsService) Update(enterpriseId string, userId string, deviceId string, installId string, install *Install) *InstallsUpdateCall {
	c := &InstallsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.installId = installId
	c.install = install
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstallsUpdateCall) Fields(s ...googleapi.Field) *InstallsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstallsUpdateCall) Context(ctx context.Context) *InstallsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstallsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstallsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.install)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.installs.update" call.
// Exactly one of *Install or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Install.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InstallsUpdateCall) Do(opts ...googleapi.CallOption) (*Install, error) {
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
	ret := &Install{
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
	//   "description": "Requests to install the latest version of an app to a device. If the app is already installed, then it is updated to the latest version if necessary.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.installs.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "installId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "installId": {
	//       "description": "The ID of the product represented by the install, e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}",
	//   "request": {
	//     "$ref": "Install"
	//   },
	//   "response": {
	//     "$ref": "Install"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsfordevice.delete":

type ManagedconfigurationsfordeviceDeleteCall struct {
	s                               *Service
	enterpriseId                    string
	userId                          string
	deviceId                        string
	managedConfigurationForDeviceId string
	urlParams_                      gensupport.URLParams
	ctx_                            context.Context
	header_                         http.Header
}

// Delete: Removes a per-device managed configuration for an app for the
// specified device.
func (r *ManagedconfigurationsfordeviceService) Delete(enterpriseId string, userId string, deviceId string, managedConfigurationForDeviceId string) *ManagedconfigurationsfordeviceDeleteCall {
	c := &ManagedconfigurationsfordeviceDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.managedConfigurationForDeviceId = managedConfigurationForDeviceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsfordeviceDeleteCall) Fields(s ...googleapi.Field) *ManagedconfigurationsfordeviceDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsfordeviceDeleteCall) Context(ctx context.Context) *ManagedconfigurationsfordeviceDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsfordeviceDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsfordeviceDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                    c.enterpriseId,
		"userId":                          c.userId,
		"deviceId":                        c.deviceId,
		"managedConfigurationForDeviceId": c.managedConfigurationForDeviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsfordevice.delete" call.
func (c *ManagedconfigurationsfordeviceDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Removes a per-device managed configuration for an app for the specified device.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.managedconfigurationsfordevice.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "managedConfigurationForDeviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForDeviceId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsfordevice.get":

type ManagedconfigurationsfordeviceGetCall struct {
	s                               *Service
	enterpriseId                    string
	userId                          string
	deviceId                        string
	managedConfigurationForDeviceId string
	urlParams_                      gensupport.URLParams
	ifNoneMatch_                    string
	ctx_                            context.Context
	header_                         http.Header
}

// Get: Retrieves details of a per-device managed configuration.
func (r *ManagedconfigurationsfordeviceService) Get(enterpriseId string, userId string, deviceId string, managedConfigurationForDeviceId string) *ManagedconfigurationsfordeviceGetCall {
	c := &ManagedconfigurationsfordeviceGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.managedConfigurationForDeviceId = managedConfigurationForDeviceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsfordeviceGetCall) Fields(s ...googleapi.Field) *ManagedconfigurationsfordeviceGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManagedconfigurationsfordeviceGetCall) IfNoneMatch(entityTag string) *ManagedconfigurationsfordeviceGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsfordeviceGetCall) Context(ctx context.Context) *ManagedconfigurationsfordeviceGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsfordeviceGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsfordeviceGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                    c.enterpriseId,
		"userId":                          c.userId,
		"deviceId":                        c.deviceId,
		"managedConfigurationForDeviceId": c.managedConfigurationForDeviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsfordevice.get" call.
// Exactly one of *ManagedConfiguration or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ManagedConfiguration.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManagedconfigurationsfordeviceGetCall) Do(opts ...googleapi.CallOption) (*ManagedConfiguration, error) {
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
	ret := &ManagedConfiguration{
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
	//   "description": "Retrieves details of a per-device managed configuration.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.managedconfigurationsfordevice.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "managedConfigurationForDeviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForDeviceId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}",
	//   "response": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsfordevice.list":

type ManagedconfigurationsfordeviceListCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all the per-device managed configurations for the
// specified device. Only the ID is set.
func (r *ManagedconfigurationsfordeviceService) List(enterpriseId string, userId string, deviceId string) *ManagedconfigurationsfordeviceListCall {
	c := &ManagedconfigurationsfordeviceListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsfordeviceListCall) Fields(s ...googleapi.Field) *ManagedconfigurationsfordeviceListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManagedconfigurationsfordeviceListCall) IfNoneMatch(entityTag string) *ManagedconfigurationsfordeviceListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsfordeviceListCall) Context(ctx context.Context) *ManagedconfigurationsfordeviceListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsfordeviceListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsfordeviceListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsfordevice.list" call.
// Exactly one of *ManagedConfigurationsForDeviceListResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *ManagedConfigurationsForDeviceListResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ManagedconfigurationsfordeviceListCall) Do(opts ...googleapi.CallOption) (*ManagedConfigurationsForDeviceListResponse, error) {
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
	ret := &ManagedConfigurationsForDeviceListResponse{
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
	//   "description": "Lists all the per-device managed configurations for the specified device. Only the ID is set.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.managedconfigurationsfordevice.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice",
	//   "response": {
	//     "$ref": "ManagedConfigurationsForDeviceListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsfordevice.patch":

type ManagedconfigurationsfordevicePatchCall struct {
	s                               *Service
	enterpriseId                    string
	userId                          string
	deviceId                        string
	managedConfigurationForDeviceId string
	managedconfiguration            *ManagedConfiguration
	urlParams_                      gensupport.URLParams
	ctx_                            context.Context
	header_                         http.Header
}

// Patch: Adds or updates a per-device managed configuration for an app
// for the specified device. This method supports patch semantics.
func (r *ManagedconfigurationsfordeviceService) Patch(enterpriseId string, userId string, deviceId string, managedConfigurationForDeviceId string, managedconfiguration *ManagedConfiguration) *ManagedconfigurationsfordevicePatchCall {
	c := &ManagedconfigurationsfordevicePatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.managedConfigurationForDeviceId = managedConfigurationForDeviceId
	c.managedconfiguration = managedconfiguration
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsfordevicePatchCall) Fields(s ...googleapi.Field) *ManagedconfigurationsfordevicePatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsfordevicePatchCall) Context(ctx context.Context) *ManagedconfigurationsfordevicePatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsfordevicePatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsfordevicePatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.managedconfiguration)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                    c.enterpriseId,
		"userId":                          c.userId,
		"deviceId":                        c.deviceId,
		"managedConfigurationForDeviceId": c.managedConfigurationForDeviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsfordevice.patch" call.
// Exactly one of *ManagedConfiguration or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ManagedConfiguration.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManagedconfigurationsfordevicePatchCall) Do(opts ...googleapi.CallOption) (*ManagedConfiguration, error) {
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
	ret := &ManagedConfiguration{
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
	//   "description": "Adds or updates a per-device managed configuration for an app for the specified device. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.managedconfigurationsfordevice.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "managedConfigurationForDeviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForDeviceId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}",
	//   "request": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "response": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsfordevice.update":

type ManagedconfigurationsfordeviceUpdateCall struct {
	s                               *Service
	enterpriseId                    string
	userId                          string
	deviceId                        string
	managedConfigurationForDeviceId string
	managedconfiguration            *ManagedConfiguration
	urlParams_                      gensupport.URLParams
	ctx_                            context.Context
	header_                         http.Header
}

// Update: Adds or updates a per-device managed configuration for an app
// for the specified device.
func (r *ManagedconfigurationsfordeviceService) Update(enterpriseId string, userId string, deviceId string, managedConfigurationForDeviceId string, managedconfiguration *ManagedConfiguration) *ManagedconfigurationsfordeviceUpdateCall {
	c := &ManagedconfigurationsfordeviceUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.managedConfigurationForDeviceId = managedConfigurationForDeviceId
	c.managedconfiguration = managedconfiguration
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsfordeviceUpdateCall) Fields(s ...googleapi.Field) *ManagedconfigurationsfordeviceUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsfordeviceUpdateCall) Context(ctx context.Context) *ManagedconfigurationsfordeviceUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsfordeviceUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsfordeviceUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.managedconfiguration)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                    c.enterpriseId,
		"userId":                          c.userId,
		"deviceId":                        c.deviceId,
		"managedConfigurationForDeviceId": c.managedConfigurationForDeviceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsfordevice.update" call.
// Exactly one of *ManagedConfiguration or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ManagedConfiguration.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManagedconfigurationsfordeviceUpdateCall) Do(opts ...googleapi.CallOption) (*ManagedConfiguration, error) {
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
	ret := &ManagedConfiguration{
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
	//   "description": "Adds or updates a per-device managed configuration for an app for the specified device.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.managedconfigurationsfordevice.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "deviceId",
	//     "managedConfigurationForDeviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "The Android ID of the device.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForDeviceId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/managedConfigurationsForDevice/{managedConfigurationForDeviceId}",
	//   "request": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "response": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsforuser.delete":

type ManagedconfigurationsforuserDeleteCall struct {
	s                             *Service
	enterpriseId                  string
	userId                        string
	managedConfigurationForUserId string
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// Delete: Removes a per-user managed configuration for an app for the
// specified user.
func (r *ManagedconfigurationsforuserService) Delete(enterpriseId string, userId string, managedConfigurationForUserId string) *ManagedconfigurationsforuserDeleteCall {
	c := &ManagedconfigurationsforuserDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.managedConfigurationForUserId = managedConfigurationForUserId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsforuserDeleteCall) Fields(s ...googleapi.Field) *ManagedconfigurationsforuserDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsforuserDeleteCall) Context(ctx context.Context) *ManagedconfigurationsforuserDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsforuserDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsforuserDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                  c.enterpriseId,
		"userId":                        c.userId,
		"managedConfigurationForUserId": c.managedConfigurationForUserId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsforuser.delete" call.
func (c *ManagedconfigurationsforuserDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Removes a per-user managed configuration for an app for the specified user.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.managedconfigurationsforuser.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "managedConfigurationForUserId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForUserId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsforuser.get":

type ManagedconfigurationsforuserGetCall struct {
	s                             *Service
	enterpriseId                  string
	userId                        string
	managedConfigurationForUserId string
	urlParams_                    gensupport.URLParams
	ifNoneMatch_                  string
	ctx_                          context.Context
	header_                       http.Header
}

// Get: Retrieves details of a per-user managed configuration for an app
// for the specified user.
func (r *ManagedconfigurationsforuserService) Get(enterpriseId string, userId string, managedConfigurationForUserId string) *ManagedconfigurationsforuserGetCall {
	c := &ManagedconfigurationsforuserGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.managedConfigurationForUserId = managedConfigurationForUserId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsforuserGetCall) Fields(s ...googleapi.Field) *ManagedconfigurationsforuserGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManagedconfigurationsforuserGetCall) IfNoneMatch(entityTag string) *ManagedconfigurationsforuserGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsforuserGetCall) Context(ctx context.Context) *ManagedconfigurationsforuserGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsforuserGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsforuserGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                  c.enterpriseId,
		"userId":                        c.userId,
		"managedConfigurationForUserId": c.managedConfigurationForUserId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsforuser.get" call.
// Exactly one of *ManagedConfiguration or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ManagedConfiguration.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManagedconfigurationsforuserGetCall) Do(opts ...googleapi.CallOption) (*ManagedConfiguration, error) {
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
	ret := &ManagedConfiguration{
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
	//   "description": "Retrieves details of a per-user managed configuration for an app for the specified user.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.managedconfigurationsforuser.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "managedConfigurationForUserId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForUserId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}",
	//   "response": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsforuser.list":

type ManagedconfigurationsforuserListCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all the per-user managed configurations for the specified
// user. Only the ID is set.
func (r *ManagedconfigurationsforuserService) List(enterpriseId string, userId string) *ManagedconfigurationsforuserListCall {
	c := &ManagedconfigurationsforuserListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsforuserListCall) Fields(s ...googleapi.Field) *ManagedconfigurationsforuserListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManagedconfigurationsforuserListCall) IfNoneMatch(entityTag string) *ManagedconfigurationsforuserListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsforuserListCall) Context(ctx context.Context) *ManagedconfigurationsforuserListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsforuserListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsforuserListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsforuser.list" call.
// Exactly one of *ManagedConfigurationsForUserListResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *ManagedConfigurationsForUserListResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ManagedconfigurationsforuserListCall) Do(opts ...googleapi.CallOption) (*ManagedConfigurationsForUserListResponse, error) {
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
	ret := &ManagedConfigurationsForUserListResponse{
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
	//   "description": "Lists all the per-user managed configurations for the specified user. Only the ID is set.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.managedconfigurationsforuser.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser",
	//   "response": {
	//     "$ref": "ManagedConfigurationsForUserListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsforuser.patch":

type ManagedconfigurationsforuserPatchCall struct {
	s                             *Service
	enterpriseId                  string
	userId                        string
	managedConfigurationForUserId string
	managedconfiguration          *ManagedConfiguration
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// Patch: Adds or updates a per-user managed configuration for an app
// for the specified user. This method supports patch semantics.
func (r *ManagedconfigurationsforuserService) Patch(enterpriseId string, userId string, managedConfigurationForUserId string, managedconfiguration *ManagedConfiguration) *ManagedconfigurationsforuserPatchCall {
	c := &ManagedconfigurationsforuserPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.managedConfigurationForUserId = managedConfigurationForUserId
	c.managedconfiguration = managedconfiguration
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsforuserPatchCall) Fields(s ...googleapi.Field) *ManagedconfigurationsforuserPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsforuserPatchCall) Context(ctx context.Context) *ManagedconfigurationsforuserPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsforuserPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsforuserPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.managedconfiguration)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                  c.enterpriseId,
		"userId":                        c.userId,
		"managedConfigurationForUserId": c.managedConfigurationForUserId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsforuser.patch" call.
// Exactly one of *ManagedConfiguration or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ManagedConfiguration.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManagedconfigurationsforuserPatchCall) Do(opts ...googleapi.CallOption) (*ManagedConfiguration, error) {
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
	ret := &ManagedConfiguration{
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
	//   "description": "Adds or updates a per-user managed configuration for an app for the specified user. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.managedconfigurationsforuser.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "managedConfigurationForUserId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForUserId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}",
	//   "request": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "response": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.managedconfigurationsforuser.update":

type ManagedconfigurationsforuserUpdateCall struct {
	s                             *Service
	enterpriseId                  string
	userId                        string
	managedConfigurationForUserId string
	managedconfiguration          *ManagedConfiguration
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// Update: Adds or updates a per-user managed configuration for an app
// for the specified user.
func (r *ManagedconfigurationsforuserService) Update(enterpriseId string, userId string, managedConfigurationForUserId string, managedconfiguration *ManagedConfiguration) *ManagedconfigurationsforuserUpdateCall {
	c := &ManagedconfigurationsforuserUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.managedConfigurationForUserId = managedConfigurationForUserId
	c.managedconfiguration = managedconfiguration
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManagedconfigurationsforuserUpdateCall) Fields(s ...googleapi.Field) *ManagedconfigurationsforuserUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManagedconfigurationsforuserUpdateCall) Context(ctx context.Context) *ManagedconfigurationsforuserUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ManagedconfigurationsforuserUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ManagedconfigurationsforuserUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.managedconfiguration)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":                  c.enterpriseId,
		"userId":                        c.userId,
		"managedConfigurationForUserId": c.managedConfigurationForUserId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.managedconfigurationsforuser.update" call.
// Exactly one of *ManagedConfiguration or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ManagedConfiguration.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManagedconfigurationsforuserUpdateCall) Do(opts ...googleapi.CallOption) (*ManagedConfiguration, error) {
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
	ret := &ManagedConfiguration{
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
	//   "description": "Adds or updates a per-user managed configuration for an app for the specified user.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.managedconfigurationsforuser.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId",
	//     "managedConfigurationForUserId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "managedConfigurationForUserId": {
	//       "description": "The ID of the managed configuration (a product ID), e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/managedConfigurationsForUser/{managedConfigurationForUserId}",
	//   "request": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "response": {
	//     "$ref": "ManagedConfiguration"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.permissions.get":

type PermissionsGetCall struct {
	s            *Service
	permissionId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves details of an Android app permission for display to an
// enterprise admin.
func (r *PermissionsService) Get(permissionId string) *PermissionsGetCall {
	c := &PermissionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.permissionId = permissionId
	return c
}

// Language sets the optional parameter "language": The BCP47 tag for
// the user's preferred language (e.g. "en-US", "de")
func (c *PermissionsGetCall) Language(language string) *PermissionsGetCall {
	c.urlParams_.Set("language", language)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PermissionsGetCall) Fields(s ...googleapi.Field) *PermissionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PermissionsGetCall) IfNoneMatch(entityTag string) *PermissionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PermissionsGetCall) Context(ctx context.Context) *PermissionsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PermissionsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PermissionsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "permissions/{permissionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"permissionId": c.permissionId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.permissions.get" call.
// Exactly one of *Permission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Permission.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PermissionsGetCall) Do(opts ...googleapi.CallOption) (*Permission, error) {
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
	ret := &Permission{
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
	//   "description": "Retrieves details of an Android app permission for display to an enterprise admin.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.permissions.get",
	//   "parameterOrder": [
	//     "permissionId"
	//   ],
	//   "parameters": {
	//     "language": {
	//       "description": "The BCP47 tag for the user's preferred language (e.g. \"en-US\", \"de\")",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "permissionId": {
	//       "description": "The ID of the permission.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "permissions/{permissionId}",
	//   "response": {
	//     "$ref": "Permission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.products.approve":

type ProductsApproveCall struct {
	s                      *Service
	enterpriseId           string
	productId              string
	productsapproverequest *ProductsApproveRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Approve: Approves the specified product and the relevant app
// permissions, if any. The maximum number of products that you can
// approve per enterprise customer is 1,000.
//
// To learn how to use managed Google Play to design and create a store
// layout to display approved products to your users, see Store Layout
// Design.
func (r *ProductsService) Approve(enterpriseId string, productId string, productsapproverequest *ProductsApproveRequest) *ProductsApproveCall {
	c := &ProductsApproveCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	c.productsapproverequest = productsapproverequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsApproveCall) Fields(s ...googleapi.Field) *ProductsApproveCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsApproveCall) Context(ctx context.Context) *ProductsApproveCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsApproveCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsApproveCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productsapproverequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/approve")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.products.approve" call.
func (c *ProductsApproveCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Approves the specified product and the relevant app permissions, if any. The maximum number of products that you can approve per enterprise customer is 1,000.\n\nTo learn how to use managed Google Play to design and create a store layout to display approved products to your users, see Store Layout Design.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.products.approve",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/products/{productId}/approve",
	//   "request": {
	//     "$ref": "ProductsApproveRequest"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.products.generateApprovalUrl":

type ProductsGenerateApprovalUrlCall struct {
	s            *Service
	enterpriseId string
	productId    string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// GenerateApprovalUrl: Generates a URL that can be rendered in an
// iframe to display the permissions (if any) of a product. An
// enterprise admin must view these permissions and accept them on
// behalf of their organization in order to approve that
// product.
//
// Admins should accept the displayed permissions by interacting with a
// separate UI element in the EMM console, which in turn should trigger
// the use of this URL as the approvalUrlInfo.approvalUrl property in a
// Products.approve call to approve the product. This URL can only be
// used to display permissions for up to 1 day.
func (r *ProductsService) GenerateApprovalUrl(enterpriseId string, productId string) *ProductsGenerateApprovalUrlCall {
	c := &ProductsGenerateApprovalUrlCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	return c
}

// LanguageCode sets the optional parameter "languageCode": The BCP 47
// language code used for permission names and descriptions in the
// returned iframe, for instance "en-US".
func (c *ProductsGenerateApprovalUrlCall) LanguageCode(languageCode string) *ProductsGenerateApprovalUrlCall {
	c.urlParams_.Set("languageCode", languageCode)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsGenerateApprovalUrlCall) Fields(s ...googleapi.Field) *ProductsGenerateApprovalUrlCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsGenerateApprovalUrlCall) Context(ctx context.Context) *ProductsGenerateApprovalUrlCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsGenerateApprovalUrlCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsGenerateApprovalUrlCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/generateApprovalUrl")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.products.generateApprovalUrl" call.
// Exactly one of *ProductsGenerateApprovalUrlResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ProductsGenerateApprovalUrlResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProductsGenerateApprovalUrlCall) Do(opts ...googleapi.CallOption) (*ProductsGenerateApprovalUrlResponse, error) {
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
	ret := &ProductsGenerateApprovalUrlResponse{
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
	//   "description": "Generates a URL that can be rendered in an iframe to display the permissions (if any) of a product. An enterprise admin must view these permissions and accept them on behalf of their organization in order to approve that product.\n\nAdmins should accept the displayed permissions by interacting with a separate UI element in the EMM console, which in turn should trigger the use of this URL as the approvalUrlInfo.approvalUrl property in a Products.approve call to approve the product. This URL can only be used to display permissions for up to 1 day.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.products.generateApprovalUrl",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "languageCode": {
	//       "description": "The BCP 47 language code used for permission names and descriptions in the returned iframe, for instance \"en-US\".",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/products/{productId}/generateApprovalUrl",
	//   "response": {
	//     "$ref": "ProductsGenerateApprovalUrlResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.products.get":

type ProductsGetCall struct {
	s            *Service
	enterpriseId string
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves details of a product for display to an enterprise
// admin.
func (r *ProductsService) Get(enterpriseId string, productId string) *ProductsGetCall {
	c := &ProductsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	return c
}

// Language sets the optional parameter "language": The BCP47 tag for
// the user's preferred language (e.g. "en-US", "de").
func (c *ProductsGetCall) Language(language string) *ProductsGetCall {
	c.urlParams_.Set("language", language)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsGetCall) Fields(s ...googleapi.Field) *ProductsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductsGetCall) IfNoneMatch(entityTag string) *ProductsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsGetCall) Context(ctx context.Context) *ProductsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.products.get" call.
// Exactly one of *Product or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Product.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProductsGetCall) Do(opts ...googleapi.CallOption) (*Product, error) {
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
	ret := &Product{
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
	//   "description": "Retrieves details of a product for display to an enterprise admin.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.products.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The BCP47 tag for the user's preferred language (e.g. \"en-US\", \"de\").",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product, e.g. \"app:com.google.android.gm\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/products/{productId}",
	//   "response": {
	//     "$ref": "Product"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.products.getAppRestrictionsSchema":

type ProductsGetAppRestrictionsSchemaCall struct {
	s            *Service
	enterpriseId string
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetAppRestrictionsSchema: Retrieves the schema that defines the
// configurable properties for this product. All products have a schema,
// but this schema may be empty if no managed configurations have been
// defined. This schema can be used to populate a UI that allows an
// admin to configure the product. To apply a managed configuration
// based on the schema obtained using this API, see Managed
// Configurations through Play.
func (r *ProductsService) GetAppRestrictionsSchema(enterpriseId string, productId string) *ProductsGetAppRestrictionsSchemaCall {
	c := &ProductsGetAppRestrictionsSchemaCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	return c
}

// Language sets the optional parameter "language": The BCP47 tag for
// the user's preferred language (e.g. "en-US", "de").
func (c *ProductsGetAppRestrictionsSchemaCall) Language(language string) *ProductsGetAppRestrictionsSchemaCall {
	c.urlParams_.Set("language", language)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsGetAppRestrictionsSchemaCall) Fields(s ...googleapi.Field) *ProductsGetAppRestrictionsSchemaCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductsGetAppRestrictionsSchemaCall) IfNoneMatch(entityTag string) *ProductsGetAppRestrictionsSchemaCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsGetAppRestrictionsSchemaCall) Context(ctx context.Context) *ProductsGetAppRestrictionsSchemaCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsGetAppRestrictionsSchemaCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsGetAppRestrictionsSchemaCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/appRestrictionsSchema")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.products.getAppRestrictionsSchema" call.
// Exactly one of *AppRestrictionsSchema or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AppRestrictionsSchema.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsGetAppRestrictionsSchemaCall) Do(opts ...googleapi.CallOption) (*AppRestrictionsSchema, error) {
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
	ret := &AppRestrictionsSchema{
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
	//   "description": "Retrieves the schema that defines the configurable properties for this product. All products have a schema, but this schema may be empty if no managed configurations have been defined. This schema can be used to populate a UI that allows an admin to configure the product. To apply a managed configuration based on the schema obtained using this API, see Managed Configurations through Play.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.products.getAppRestrictionsSchema",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The BCP47 tag for the user's preferred language (e.g. \"en-US\", \"de\").",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/products/{productId}/appRestrictionsSchema",
	//   "response": {
	//     "$ref": "AppRestrictionsSchema"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.products.getPermissions":

type ProductsGetPermissionsCall struct {
	s            *Service
	enterpriseId string
	productId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetPermissions: Retrieves the Android app permissions required by
// this app.
func (r *ProductsService) GetPermissions(enterpriseId string, productId string) *ProductsGetPermissionsCall {
	c := &ProductsGetPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsGetPermissionsCall) Fields(s ...googleapi.Field) *ProductsGetPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductsGetPermissionsCall) IfNoneMatch(entityTag string) *ProductsGetPermissionsCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsGetPermissionsCall) Context(ctx context.Context) *ProductsGetPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsGetPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsGetPermissionsCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/permissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.products.getPermissions" call.
// Exactly one of *ProductPermissions or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ProductPermissions.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsGetPermissionsCall) Do(opts ...googleapi.CallOption) (*ProductPermissions, error) {
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
	ret := &ProductPermissions{
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
	//   "description": "Retrieves the Android app permissions required by this app.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.products.getPermissions",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/products/{productId}/permissions",
	//   "response": {
	//     "$ref": "ProductPermissions"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.products.list":

type ProductsListCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Finds approved products that match a query, or all approved
// products if there is no query.
func (r *ProductsService) List(enterpriseId string) *ProductsListCall {
	c := &ProductsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Approved sets the optional parameter "approved": Specifies whether to
// search among all products (false) or among only products that have
// been approved (true). Only "true" is supported, and should be
// specified.
func (c *ProductsListCall) Approved(approved bool) *ProductsListCall {
	c.urlParams_.Set("approved", fmt.Sprint(approved))
	return c
}

// Language sets the optional parameter "language": The BCP47 tag for
// the user's preferred language (e.g. "en-US", "de"). Results are
// returned in the language best matching the preferred language.
func (c *ProductsListCall) Language(language string) *ProductsListCall {
	c.urlParams_.Set("language", language)
	return c
}

// MaxResults sets the optional parameter "maxResults": Specifies the
// maximum number of products that can be returned per request. If not
// specified, uses a default value of 100, which is also the maximum
// retrievable within a single response.
func (c *ProductsListCall) MaxResults(maxResults int64) *ProductsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// Query sets the optional parameter "query": The search query as typed
// in the Google Play store search box. If omitted, all approved apps
// will be returned (using the pagination parameters), including apps
// that are not available in the store (e.g. unpublished apps).
func (c *ProductsListCall) Query(query string) *ProductsListCall {
	c.urlParams_.Set("query", query)
	return c
}

// Token sets the optional parameter "token": A pagination token is
// contained in a request''s response when there are more products. The
// token can be used in a subsequent request to obtain more products,
// and so forth. This parameter cannot be used in the initial request.
func (c *ProductsListCall) Token(token string) *ProductsListCall {
	c.urlParams_.Set("token", token)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsListCall) Fields(s ...googleapi.Field) *ProductsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProductsListCall) IfNoneMatch(entityTag string) *ProductsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsListCall) Context(ctx context.Context) *ProductsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.products.list" call.
// Exactly one of *ProductsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ProductsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsListCall) Do(opts ...googleapi.CallOption) (*ProductsListResponse, error) {
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
	ret := &ProductsListResponse{
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
	//   "description": "Finds approved products that match a query, or all approved products if there is no query.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.products.list",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "approved": {
	//       "description": "Specifies whether to search among all products (false) or among only products that have been approved (true). Only \"true\" is supported, and should be specified.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "language": {
	//       "description": "The BCP47 tag for the user's preferred language (e.g. \"en-US\", \"de\"). Results are returned in the language best matching the preferred language.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Specifies the maximum number of products that can be returned per request. If not specified, uses a default value of 100, which is also the maximum retrievable within a single response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "query": {
	//       "description": "The search query as typed in the Google Play store search box. If omitted, all approved apps will be returned (using the pagination parameters), including apps that are not available in the store (e.g. unpublished apps).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "token": {
	//       "description": "A pagination token is contained in a request''s response when there are more products. The token can be used in a subsequent request to obtain more products, and so forth. This parameter cannot be used in the initial request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/products",
	//   "response": {
	//     "$ref": "ProductsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.products.unapprove":

type ProductsUnapproveCall struct {
	s            *Service
	enterpriseId string
	productId    string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Unapprove: Unapproves the specified product (and the relevant app
// permissions, if any)
func (r *ProductsService) Unapprove(enterpriseId string, productId string) *ProductsUnapproveCall {
	c := &ProductsUnapproveCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsUnapproveCall) Fields(s ...googleapi.Field) *ProductsUnapproveCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsUnapproveCall) Context(ctx context.Context) *ProductsUnapproveCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProductsUnapproveCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProductsUnapproveCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/unapprove")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.products.unapprove" call.
func (c *ProductsUnapproveCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Unapproves the specified product (and the relevant app permissions, if any)",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.products.unapprove",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "The ID of the product.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/products/{productId}/unapprove",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.serviceaccountkeys.delete":

type ServiceaccountkeysDeleteCall struct {
	s            *Service
	enterpriseId string
	keyId        string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Removes and invalidates the specified credentials for the
// service account associated with this enterprise. The calling service
// account must have been retrieved by calling
// Enterprises.GetServiceAccount and must have been set as the
// enterprise service account by calling Enterprises.SetAccount.
func (r *ServiceaccountkeysService) Delete(enterpriseId string, keyId string) *ServiceaccountkeysDeleteCall {
	c := &ServiceaccountkeysDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.keyId = keyId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServiceaccountkeysDeleteCall) Fields(s ...googleapi.Field) *ServiceaccountkeysDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServiceaccountkeysDeleteCall) Context(ctx context.Context) *ServiceaccountkeysDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServiceaccountkeysDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServiceaccountkeysDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/serviceAccountKeys/{keyId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"keyId":        c.keyId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.serviceaccountkeys.delete" call.
func (c *ServiceaccountkeysDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Removes and invalidates the specified credentials for the service account associated with this enterprise. The calling service account must have been retrieved by calling Enterprises.GetServiceAccount and must have been set as the enterprise service account by calling Enterprises.SetAccount.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.serviceaccountkeys.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "keyId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "keyId": {
	//       "description": "The ID of the key.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/serviceAccountKeys/{keyId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.serviceaccountkeys.insert":

type ServiceaccountkeysInsertCall struct {
	s                 *Service
	enterpriseId      string
	serviceaccountkey *ServiceAccountKey
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Insert: Generates new credentials for the service account associated
// with this enterprise. The calling service account must have been
// retrieved by calling Enterprises.GetServiceAccount and must have been
// set as the enterprise service account by calling
// Enterprises.SetAccount.
//
// Only the type of the key should be populated in the resource to be
// inserted.
func (r *ServiceaccountkeysService) Insert(enterpriseId string, serviceaccountkey *ServiceAccountKey) *ServiceaccountkeysInsertCall {
	c := &ServiceaccountkeysInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.serviceaccountkey = serviceaccountkey
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServiceaccountkeysInsertCall) Fields(s ...googleapi.Field) *ServiceaccountkeysInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServiceaccountkeysInsertCall) Context(ctx context.Context) *ServiceaccountkeysInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServiceaccountkeysInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServiceaccountkeysInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.serviceaccountkey)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/serviceAccountKeys")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.serviceaccountkeys.insert" call.
// Exactly one of *ServiceAccountKey or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ServiceAccountKey.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServiceaccountkeysInsertCall) Do(opts ...googleapi.CallOption) (*ServiceAccountKey, error) {
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
	ret := &ServiceAccountKey{
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
	//   "description": "Generates new credentials for the service account associated with this enterprise. The calling service account must have been retrieved by calling Enterprises.GetServiceAccount and must have been set as the enterprise service account by calling Enterprises.SetAccount.\n\nOnly the type of the key should be populated in the resource to be inserted.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.serviceaccountkeys.insert",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/serviceAccountKeys",
	//   "request": {
	//     "$ref": "ServiceAccountKey"
	//   },
	//   "response": {
	//     "$ref": "ServiceAccountKey"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.serviceaccountkeys.list":

type ServiceaccountkeysListCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all active credentials for the service account associated
// with this enterprise. Only the ID and key type are returned. The
// calling service account must have been retrieved by calling
// Enterprises.GetServiceAccount and must have been set as the
// enterprise service account by calling Enterprises.SetAccount.
func (r *ServiceaccountkeysService) List(enterpriseId string) *ServiceaccountkeysListCall {
	c := &ServiceaccountkeysListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServiceaccountkeysListCall) Fields(s ...googleapi.Field) *ServiceaccountkeysListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ServiceaccountkeysListCall) IfNoneMatch(entityTag string) *ServiceaccountkeysListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServiceaccountkeysListCall) Context(ctx context.Context) *ServiceaccountkeysListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServiceaccountkeysListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServiceaccountkeysListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/serviceAccountKeys")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.serviceaccountkeys.list" call.
// Exactly one of *ServiceAccountKeysListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ServiceAccountKeysListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServiceaccountkeysListCall) Do(opts ...googleapi.CallOption) (*ServiceAccountKeysListResponse, error) {
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
	ret := &ServiceAccountKeysListResponse{
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
	//   "description": "Lists all active credentials for the service account associated with this enterprise. Only the ID and key type are returned. The calling service account must have been retrieved by calling Enterprises.GetServiceAccount and must have been set as the enterprise service account by calling Enterprises.SetAccount.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.serviceaccountkeys.list",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/serviceAccountKeys",
	//   "response": {
	//     "$ref": "ServiceAccountKeysListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutclusters.delete":

type StorelayoutclustersDeleteCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	clusterId    string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deletes a cluster.
func (r *StorelayoutclustersService) Delete(enterpriseId string, pageId string, clusterId string) *StorelayoutclustersDeleteCall {
	c := &StorelayoutclustersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.clusterId = clusterId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutclustersDeleteCall) Fields(s ...googleapi.Field) *StorelayoutclustersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutclustersDeleteCall) Context(ctx context.Context) *StorelayoutclustersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutclustersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutclustersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutclusters.delete" call.
func (c *StorelayoutclustersDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes a cluster.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.storelayoutclusters.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The ID of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutclusters.get":

type StorelayoutclustersGetCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	clusterId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves details of a cluster.
func (r *StorelayoutclustersService) Get(enterpriseId string, pageId string, clusterId string) *StorelayoutclustersGetCall {
	c := &StorelayoutclustersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.clusterId = clusterId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutclustersGetCall) Fields(s ...googleapi.Field) *StorelayoutclustersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *StorelayoutclustersGetCall) IfNoneMatch(entityTag string) *StorelayoutclustersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutclustersGetCall) Context(ctx context.Context) *StorelayoutclustersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutclustersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutclustersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutclusters.get" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersGetCall) Do(opts ...googleapi.CallOption) (*StoreCluster, error) {
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
	ret := &StoreCluster{
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
	//   "description": "Retrieves details of a cluster.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.storelayoutclusters.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The ID of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}",
	//   "response": {
	//     "$ref": "StoreCluster"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutclusters.insert":

type StorelayoutclustersInsertCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	storecluster *StoreCluster
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Insert: Inserts a new cluster in a page.
func (r *StorelayoutclustersService) Insert(enterpriseId string, pageId string, storecluster *StoreCluster) *StorelayoutclustersInsertCall {
	c := &StorelayoutclustersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.storecluster = storecluster
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutclustersInsertCall) Fields(s ...googleapi.Field) *StorelayoutclustersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutclustersInsertCall) Context(ctx context.Context) *StorelayoutclustersInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutclustersInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutclustersInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storecluster)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutclusters.insert" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersInsertCall) Do(opts ...googleapi.CallOption) (*StoreCluster, error) {
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
	ret := &StoreCluster{
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
	//   "description": "Inserts a new cluster in a page.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.storelayoutclusters.insert",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters",
	//   "request": {
	//     "$ref": "StoreCluster"
	//   },
	//   "response": {
	//     "$ref": "StoreCluster"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutclusters.list":

type StorelayoutclustersListCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the details of all clusters on the specified page.
func (r *StorelayoutclustersService) List(enterpriseId string, pageId string) *StorelayoutclustersListCall {
	c := &StorelayoutclustersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutclustersListCall) Fields(s ...googleapi.Field) *StorelayoutclustersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *StorelayoutclustersListCall) IfNoneMatch(entityTag string) *StorelayoutclustersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutclustersListCall) Context(ctx context.Context) *StorelayoutclustersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutclustersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutclustersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutclusters.list" call.
// Exactly one of *StoreLayoutClustersListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *StoreLayoutClustersListResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *StorelayoutclustersListCall) Do(opts ...googleapi.CallOption) (*StoreLayoutClustersListResponse, error) {
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
	ret := &StoreLayoutClustersListResponse{
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
	//   "description": "Retrieves the details of all clusters on the specified page.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.storelayoutclusters.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters",
	//   "response": {
	//     "$ref": "StoreLayoutClustersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutclusters.patch":

type StorelayoutclustersPatchCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	clusterId    string
	storecluster *StoreCluster
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Patch: Updates a cluster. This method supports patch semantics.
func (r *StorelayoutclustersService) Patch(enterpriseId string, pageId string, clusterId string, storecluster *StoreCluster) *StorelayoutclustersPatchCall {
	c := &StorelayoutclustersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.clusterId = clusterId
	c.storecluster = storecluster
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutclustersPatchCall) Fields(s ...googleapi.Field) *StorelayoutclustersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutclustersPatchCall) Context(ctx context.Context) *StorelayoutclustersPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutclustersPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutclustersPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storecluster)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutclusters.patch" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersPatchCall) Do(opts ...googleapi.CallOption) (*StoreCluster, error) {
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
	ret := &StoreCluster{
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
	//   "description": "Updates a cluster. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.storelayoutclusters.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The ID of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}",
	//   "request": {
	//     "$ref": "StoreCluster"
	//   },
	//   "response": {
	//     "$ref": "StoreCluster"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutclusters.update":

type StorelayoutclustersUpdateCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	clusterId    string
	storecluster *StoreCluster
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Updates a cluster.
func (r *StorelayoutclustersService) Update(enterpriseId string, pageId string, clusterId string, storecluster *StoreCluster) *StorelayoutclustersUpdateCall {
	c := &StorelayoutclustersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.clusterId = clusterId
	c.storecluster = storecluster
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutclustersUpdateCall) Fields(s ...googleapi.Field) *StorelayoutclustersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutclustersUpdateCall) Context(ctx context.Context) *StorelayoutclustersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutclustersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutclustersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storecluster)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutclusters.update" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersUpdateCall) Do(opts ...googleapi.CallOption) (*StoreCluster, error) {
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
	ret := &StoreCluster{
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
	//   "description": "Updates a cluster.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.storelayoutclusters.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The ID of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}",
	//   "request": {
	//     "$ref": "StoreCluster"
	//   },
	//   "response": {
	//     "$ref": "StoreCluster"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutpages.delete":

type StorelayoutpagesDeleteCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deletes a store page.
func (r *StorelayoutpagesService) Delete(enterpriseId string, pageId string) *StorelayoutpagesDeleteCall {
	c := &StorelayoutpagesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutpagesDeleteCall) Fields(s ...googleapi.Field) *StorelayoutpagesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutpagesDeleteCall) Context(ctx context.Context) *StorelayoutpagesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutpagesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutpagesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutpages.delete" call.
func (c *StorelayoutpagesDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes a store page.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.storelayoutpages.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutpages.get":

type StorelayoutpagesGetCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves details of a store page.
func (r *StorelayoutpagesService) Get(enterpriseId string, pageId string) *StorelayoutpagesGetCall {
	c := &StorelayoutpagesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutpagesGetCall) Fields(s ...googleapi.Field) *StorelayoutpagesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *StorelayoutpagesGetCall) IfNoneMatch(entityTag string) *StorelayoutpagesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutpagesGetCall) Context(ctx context.Context) *StorelayoutpagesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutpagesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutpagesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutpages.get" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesGetCall) Do(opts ...googleapi.CallOption) (*StorePage, error) {
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
	ret := &StorePage{
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
	//   "description": "Retrieves details of a store page.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.storelayoutpages.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}",
	//   "response": {
	//     "$ref": "StorePage"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutpages.insert":

type StorelayoutpagesInsertCall struct {
	s            *Service
	enterpriseId string
	storepage    *StorePage
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Insert: Inserts a new store page.
func (r *StorelayoutpagesService) Insert(enterpriseId string, storepage *StorePage) *StorelayoutpagesInsertCall {
	c := &StorelayoutpagesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.storepage = storepage
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutpagesInsertCall) Fields(s ...googleapi.Field) *StorelayoutpagesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutpagesInsertCall) Context(ctx context.Context) *StorelayoutpagesInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutpagesInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutpagesInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storepage)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutpages.insert" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesInsertCall) Do(opts ...googleapi.CallOption) (*StorePage, error) {
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
	ret := &StorePage{
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
	//   "description": "Inserts a new store page.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.storelayoutpages.insert",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages",
	//   "request": {
	//     "$ref": "StorePage"
	//   },
	//   "response": {
	//     "$ref": "StorePage"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutpages.list":

type StorelayoutpagesListCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the details of all pages in the store.
func (r *StorelayoutpagesService) List(enterpriseId string) *StorelayoutpagesListCall {
	c := &StorelayoutpagesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutpagesListCall) Fields(s ...googleapi.Field) *StorelayoutpagesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *StorelayoutpagesListCall) IfNoneMatch(entityTag string) *StorelayoutpagesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutpagesListCall) Context(ctx context.Context) *StorelayoutpagesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutpagesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutpagesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutpages.list" call.
// Exactly one of *StoreLayoutPagesListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *StoreLayoutPagesListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *StorelayoutpagesListCall) Do(opts ...googleapi.CallOption) (*StoreLayoutPagesListResponse, error) {
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
	ret := &StoreLayoutPagesListResponse{
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
	//   "description": "Retrieves the details of all pages in the store.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.storelayoutpages.list",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages",
	//   "response": {
	//     "$ref": "StoreLayoutPagesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutpages.patch":

type StorelayoutpagesPatchCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	storepage    *StorePage
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Patch: Updates the content of a store page. This method supports
// patch semantics.
func (r *StorelayoutpagesService) Patch(enterpriseId string, pageId string, storepage *StorePage) *StorelayoutpagesPatchCall {
	c := &StorelayoutpagesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.storepage = storepage
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutpagesPatchCall) Fields(s ...googleapi.Field) *StorelayoutpagesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutpagesPatchCall) Context(ctx context.Context) *StorelayoutpagesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutpagesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutpagesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storepage)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutpages.patch" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesPatchCall) Do(opts ...googleapi.CallOption) (*StorePage, error) {
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
	ret := &StorePage{
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
	//   "description": "Updates the content of a store page. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.storelayoutpages.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}",
	//   "request": {
	//     "$ref": "StorePage"
	//   },
	//   "response": {
	//     "$ref": "StorePage"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.storelayoutpages.update":

type StorelayoutpagesUpdateCall struct {
	s            *Service
	enterpriseId string
	pageId       string
	storepage    *StorePage
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Updates the content of a store page.
func (r *StorelayoutpagesService) Update(enterpriseId string, pageId string, storepage *StorePage) *StorelayoutpagesUpdateCall {
	c := &StorelayoutpagesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.storepage = storepage
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StorelayoutpagesUpdateCall) Fields(s ...googleapi.Field) *StorelayoutpagesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StorelayoutpagesUpdateCall) Context(ctx context.Context) *StorelayoutpagesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *StorelayoutpagesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *StorelayoutpagesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storepage)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.storelayoutpages.update" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesUpdateCall) Do(opts ...googleapi.CallOption) (*StorePage, error) {
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
	ret := &StorePage{
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
	//   "description": "Updates the content of a store page.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.storelayoutpages.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/storeLayout/pages/{pageId}",
	//   "request": {
	//     "$ref": "StorePage"
	//   },
	//   "response": {
	//     "$ref": "StorePage"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.delete":

type UsersDeleteCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deleted an EMM-managed user.
func (r *UsersService) Delete(enterpriseId string, userId string) *UsersDeleteCall {
	c := &UsersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDeleteCall) Fields(s ...googleapi.Field) *UsersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDeleteCall) Context(ctx context.Context) *UsersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.delete" call.
func (c *UsersDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deleted an EMM-managed user.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.users.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.generateAuthenticationToken":

type UsersGenerateAuthenticationTokenCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// GenerateAuthenticationToken: Generates an authentication token which
// the device policy client can use to provision the given EMM-managed
// user account on a device. The generated token is single-use and
// expires after a few minutes.
//
// This call only works with EMM-managed accounts.
func (r *UsersService) GenerateAuthenticationToken(enterpriseId string, userId string) *UsersGenerateAuthenticationTokenCall {
	c := &UsersGenerateAuthenticationTokenCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGenerateAuthenticationTokenCall) Fields(s ...googleapi.Field) *UsersGenerateAuthenticationTokenCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGenerateAuthenticationTokenCall) Context(ctx context.Context) *UsersGenerateAuthenticationTokenCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersGenerateAuthenticationTokenCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersGenerateAuthenticationTokenCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/authenticationToken")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.generateAuthenticationToken" call.
// Exactly one of *AuthenticationToken or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AuthenticationToken.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersGenerateAuthenticationTokenCall) Do(opts ...googleapi.CallOption) (*AuthenticationToken, error) {
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
	ret := &AuthenticationToken{
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
	//   "description": "Generates an authentication token which the device policy client can use to provision the given EMM-managed user account on a device. The generated token is single-use and expires after a few minutes.\n\nThis call only works with EMM-managed accounts.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.users.generateAuthenticationToken",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/authenticationToken",
	//   "response": {
	//     "$ref": "AuthenticationToken"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.generateToken":

type UsersGenerateTokenCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// GenerateToken: Generates a token (activation code) to allow this user
// to configure their managed account in the Android Setup Wizard.
// Revokes any previously generated token.
//
// This call only works with Google managed accounts.
func (r *UsersService) GenerateToken(enterpriseId string, userId string) *UsersGenerateTokenCall {
	c := &UsersGenerateTokenCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGenerateTokenCall) Fields(s ...googleapi.Field) *UsersGenerateTokenCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGenerateTokenCall) Context(ctx context.Context) *UsersGenerateTokenCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersGenerateTokenCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersGenerateTokenCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/token")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.generateToken" call.
// Exactly one of *UserToken or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserToken.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersGenerateTokenCall) Do(opts ...googleapi.CallOption) (*UserToken, error) {
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
	ret := &UserToken{
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
	//   "description": "Generates a token (activation code) to allow this user to configure their managed account in the Android Setup Wizard. Revokes any previously generated token.\n\nThis call only works with Google managed accounts.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.users.generateToken",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/token",
	//   "response": {
	//     "$ref": "UserToken"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.get":

type UsersGetCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a user's details.
func (r *UsersService) Get(enterpriseId string, userId string) *UsersGetCall {
	c := &UsersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGetCall) Fields(s ...googleapi.Field) *UsersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersGetCall) IfNoneMatch(entityTag string) *UsersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGetCall) Context(ctx context.Context) *UsersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.get" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *UsersGetCall) Do(opts ...googleapi.CallOption) (*User, error) {
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
	ret := &User{
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
	//   "description": "Retrieves a user's details.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.users.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}",
	//   "response": {
	//     "$ref": "User"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.getAvailableProductSet":

type UsersGetAvailableProductSetCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetAvailableProductSet: Retrieves the set of products a user is
// entitled to access.
func (r *UsersService) GetAvailableProductSet(enterpriseId string, userId string) *UsersGetAvailableProductSetCall {
	c := &UsersGetAvailableProductSetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGetAvailableProductSetCall) Fields(s ...googleapi.Field) *UsersGetAvailableProductSetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersGetAvailableProductSetCall) IfNoneMatch(entityTag string) *UsersGetAvailableProductSetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGetAvailableProductSetCall) Context(ctx context.Context) *UsersGetAvailableProductSetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersGetAvailableProductSetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersGetAvailableProductSetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/availableProductSet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.getAvailableProductSet" call.
// Exactly one of *ProductSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProductSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersGetAvailableProductSetCall) Do(opts ...googleapi.CallOption) (*ProductSet, error) {
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
	ret := &ProductSet{
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
	//   "description": "Retrieves the set of products a user is entitled to access.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.users.getAvailableProductSet",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/availableProductSet",
	//   "response": {
	//     "$ref": "ProductSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.insert":

type UsersInsertCall struct {
	s            *Service
	enterpriseId string
	user         *User
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Insert: Creates a new EMM-managed user.
//
// The Users resource passed in the body of the request should include
// an accountIdentifier and an accountType.
// If a corresponding user already exists with the same account
// identifier, the user will be updated with the resource. In this case
// only the displayName field can be changed.
func (r *UsersService) Insert(enterpriseId string, user *User) *UsersInsertCall {
	c := &UsersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersInsertCall) Fields(s ...googleapi.Field) *UsersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersInsertCall) Context(ctx context.Context) *UsersInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.insert" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *UsersInsertCall) Do(opts ...googleapi.CallOption) (*User, error) {
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
	ret := &User{
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
	//   "description": "Creates a new EMM-managed user.\n\nThe Users resource passed in the body of the request should include an accountIdentifier and an accountType.\nIf a corresponding user already exists with the same account identifier, the user will be updated with the resource. In this case only the displayName field can be changed.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.users.insert",
	//   "parameterOrder": [
	//     "enterpriseId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users",
	//   "request": {
	//     "$ref": "User"
	//   },
	//   "response": {
	//     "$ref": "User"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.list":

type UsersListCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Looks up a user by primary email address. This is only
// supported for Google-managed users. Lookup of the id is not needed
// for EMM-managed users because the id is already returned in the
// result of the Users.insert call.
func (r *UsersService) List(enterpriseId string, email string) *UsersListCall {
	c := &UsersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.urlParams_.Set("email", email)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersListCall) Fields(s ...googleapi.Field) *UsersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersListCall) IfNoneMatch(entityTag string) *UsersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersListCall) Context(ctx context.Context) *UsersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.list" call.
// Exactly one of *UsersListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *UsersListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersListCall) Do(opts ...googleapi.CallOption) (*UsersListResponse, error) {
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
	ret := &UsersListResponse{
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
	//   "description": "Looks up a user by primary email address. This is only supported for Google-managed users. Lookup of the id is not needed for EMM-managed users because the id is already returned in the result of the Users.insert call.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.users.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "email"
	//   ],
	//   "parameters": {
	//     "email": {
	//       "description": "The exact primary email address of the user to look up.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users",
	//   "response": {
	//     "$ref": "UsersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.patch":

type UsersPatchCall struct {
	s            *Service
	enterpriseId string
	userId       string
	user         *User
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Patch: Updates the details of an EMM-managed user.
//
// Can be used with EMM-managed users only (not Google managed users).
// Pass the new details in the Users resource in the request body. Only
// the displayName field can be changed. Other fields must either be
// unset or have the currently active value. This method supports patch
// semantics.
func (r *UsersService) Patch(enterpriseId string, userId string, user *User) *UsersPatchCall {
	c := &UsersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersPatchCall) Fields(s ...googleapi.Field) *UsersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersPatchCall) Context(ctx context.Context) *UsersPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.patch" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *UsersPatchCall) Do(opts ...googleapi.CallOption) (*User, error) {
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
	ret := &User{
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
	//   "description": "Updates the details of an EMM-managed user.\n\nCan be used with EMM-managed users only (not Google managed users). Pass the new details in the Users resource in the request body. Only the displayName field can be changed. Other fields must either be unset or have the currently active value. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.users.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}",
	//   "request": {
	//     "$ref": "User"
	//   },
	//   "response": {
	//     "$ref": "User"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.revokeToken":

type UsersRevokeTokenCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// RevokeToken: Revokes a previously generated token (activation code)
// for the user.
func (r *UsersService) RevokeToken(enterpriseId string, userId string) *UsersRevokeTokenCall {
	c := &UsersRevokeTokenCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersRevokeTokenCall) Fields(s ...googleapi.Field) *UsersRevokeTokenCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersRevokeTokenCall) Context(ctx context.Context) *UsersRevokeTokenCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersRevokeTokenCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersRevokeTokenCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/token")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.revokeToken" call.
func (c *UsersRevokeTokenCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Revokes a previously generated token (activation code) for the user.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.users.revokeToken",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/token",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.setAvailableProductSet":

type UsersSetAvailableProductSetCall struct {
	s            *Service
	enterpriseId string
	userId       string
	productset   *ProductSet
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// SetAvailableProductSet: Modifies the set of products that a user is
// entitled to access (referred to as whitelisted products). Only
// products that are approved or products that were previously approved
// (products with revoked approval) can be whitelisted.
func (r *UsersService) SetAvailableProductSet(enterpriseId string, userId string, productset *ProductSet) *UsersSetAvailableProductSetCall {
	c := &UsersSetAvailableProductSetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.productset = productset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersSetAvailableProductSetCall) Fields(s ...googleapi.Field) *UsersSetAvailableProductSetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersSetAvailableProductSetCall) Context(ctx context.Context) *UsersSetAvailableProductSetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersSetAvailableProductSetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersSetAvailableProductSetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/availableProductSet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.setAvailableProductSet" call.
// Exactly one of *ProductSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProductSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersSetAvailableProductSetCall) Do(opts ...googleapi.CallOption) (*ProductSet, error) {
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
	ret := &ProductSet{
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
	//   "description": "Modifies the set of products that a user is entitled to access (referred to as whitelisted products). Only products that are approved or products that were previously approved (products with revoked approval) can be whitelisted.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.users.setAvailableProductSet",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}/availableProductSet",
	//   "request": {
	//     "$ref": "ProductSet"
	//   },
	//   "response": {
	//     "$ref": "ProductSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.users.update":

type UsersUpdateCall struct {
	s            *Service
	enterpriseId string
	userId       string
	user         *User
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Updates the details of an EMM-managed user.
//
// Can be used with EMM-managed users only (not Google managed users).
// Pass the new details in the Users resource in the request body. Only
// the displayName field can be changed. Other fields must either be
// unset or have the currently active value.
func (r *UsersService) Update(enterpriseId string, userId string, user *User) *UsersUpdateCall {
	c := &UsersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersUpdateCall) Fields(s ...googleapi.Field) *UsersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersUpdateCall) Context(ctx context.Context) *UsersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidenterprise.users.update" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *UsersUpdateCall) Do(opts ...googleapi.CallOption) (*User, error) {
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
	ret := &User{
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
	//   "description": "Updates the details of an EMM-managed user.\n\nCan be used with EMM-managed users only (not Google managed users). Pass the new details in the Users resource in the request body. Only the displayName field can be changed. Other fields must either be unset or have the currently active value.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.users.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "enterpriseId": {
	//       "description": "The ID of the enterprise.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "enterprises/{enterpriseId}/users/{userId}",
	//   "request": {
	//     "$ref": "User"
	//   },
	//   "response": {
	//     "$ref": "User"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}
