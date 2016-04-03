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
	s.Collections = NewCollectionsService(s)
	s.Collectionviewers = NewCollectionviewersService(s)
	s.Devices = NewDevicesService(s)
	s.Enterprises = NewEnterprisesService(s)
	s.Entitlements = NewEntitlementsService(s)
	s.Grouplicenses = NewGrouplicensesService(s)
	s.Grouplicenseusers = NewGrouplicenseusersService(s)
	s.Installs = NewInstallsService(s)
	s.Permissions = NewPermissionsService(s)
	s.Products = NewProductsService(s)
	s.Storelayoutclusters = NewStorelayoutclustersService(s)
	s.Storelayoutpages = NewStorelayoutpagesService(s)
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Collections *CollectionsService

	Collectionviewers *CollectionviewersService

	Devices *DevicesService

	Enterprises *EnterprisesService

	Entitlements *EntitlementsService

	Grouplicenses *GrouplicensesService

	Grouplicenseusers *GrouplicenseusersService

	Installs *InstallsService

	Permissions *PermissionsService

	Products *ProductsService

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

func NewCollectionsService(s *Service) *CollectionsService {
	rs := &CollectionsService{s: s}
	return rs
}

type CollectionsService struct {
	s *Service
}

func NewCollectionviewersService(s *Service) *CollectionviewersService {
	rs := &CollectionviewersService{s: s}
	return rs
}

type CollectionviewersService struct {
	s *Service
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
}

func (s *AppRestrictionsSchema) MarshalJSON() ([]byte, error) {
	type noMethod AppRestrictionsSchema
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AppRestrictionsSchemaRestriction: A restriction in the App
// Restriction Schema represents a piece of configuration that may be
// pre-applied.
type AppRestrictionsSchemaRestriction struct {
	// DefaultValue: The default value of the restriction.
	DefaultValue *AppRestrictionsSchemaRestrictionRestrictionValue `json:"defaultValue,omitempty"`

	// Description: A longer description of the restriction, giving more
	// detail of what it affects.
	Description string `json:"description,omitempty"`

	// Entry: For choice or multiselect restrictions, the list of possible
	// entries' human-readable names.
	Entry []string `json:"entry,omitempty"`

	// EntryValue: For choice or multiselect restrictions, the list of
	// possible entries' machine-readable values.
	EntryValue []string `json:"entryValue,omitempty"`

	// Key: The unique key that the product uses to identify the
	// restriction, e.g. "com.google.android.gm.fieldname".
	Key string `json:"key,omitempty"`

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
}

func (s *AppRestrictionsSchemaRestriction) MarshalJSON() ([]byte, error) {
	type noMethod AppRestrictionsSchemaRestriction
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *AppRestrictionsSchemaRestrictionRestrictionValue) MarshalJSON() ([]byte, error) {
	type noMethod AppRestrictionsSchemaRestrictionRestrictionValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AppVersion: This represents a single version of the app.
type AppVersion struct {
	// VersionCode: Unique increasing identifier for the app version.
	VersionCode int64 `json:"versionCode,omitempty"`

	// VersionString: The string used in the Play Store by the app developer
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
}

func (s *AppVersion) MarshalJSON() ([]byte, error) {
	type noMethod AppVersion
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ApprovalUrlInfo) MarshalJSON() ([]byte, error) {
	type noMethod ApprovalUrlInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Collection: A collection resource defines a named set of apps that is
// visible to a set of users in the Google Play Store app running on
// those users' managed devices. Those users can then install any of
// those apps if they wish (which will trigger creation of install and
// entitlement resources). A user cannot install an app on a managed
// device unless the app is listed in at least one collection that is
// visible to that user.
//
// Note that the API can be used to directly install an app regardless
// of whether it is in any collection - so an enterprise has a choice of
// either directly pushing apps to users, or allowing users to install
// apps if they want. Which is appropriate will depend on the
// enterprise's policies and the purpose of the apps concerned.
type Collection struct {
	// CollectionId: Arbitrary unique ID, allocated by the API on creation.
	CollectionId string `json:"collectionId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#collection".
	Kind string `json:"kind,omitempty"`

	// Name: A user-friendly name for the collection (should be unique),
	// e.g. "Accounting apps".
	Name string `json:"name,omitempty"`

	// ProductId: The IDs of the products in the collection, in the order in
	// which they should be displayed.
	ProductId []string `json:"productId,omitempty"`

	// Visibility: Whether this collection is visible to all users, or only
	// to the users that have been granted access through the
	// "Collectionviewers" API. With the launch of the
	// "setAvailableProductSet" API, this property should always be set to
	// "viewersOnly", as the "allUsers" option will bypass the
	// "availableProductSet" for all users within a domain.
	//
	// The "allUsers" setting is deprecated, and will be removed.
	Visibility string `json:"visibility,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CollectionId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Collection) MarshalJSON() ([]byte, error) {
	type noMethod Collection
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CollectionViewersListResponse: The user resources for the collection.
type CollectionViewersListResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#collectionViewersListResponse".
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
}

func (s *CollectionViewersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod CollectionViewersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CollectionsListResponse: The collection resources for the enterprise.
type CollectionsListResponse struct {
	// Collection: An ordered collection of products which can be made
	// visible on the Google Play Store to a selected group of users.
	Collection []*Collection `json:"collection,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#collectionsListResponse".
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Collection") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CollectionsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod CollectionsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Device: A device resource represents a mobile device managed by the
// MDM and belonging to a specific enterprise user.
//
// This collection cannot be modified via the API; it is automatically
// populated as devices are set up to be managed.
type Device struct {
	// AndroidId: The Google Play Services Android ID for the device encoded
	// as a lowercase hex string, e.g. "123456789abcdef0".
	AndroidId string `json:"androidId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#device".
	Kind string `json:"kind,omitempty"`

	// ManagementType: The mechanism by which this device is managed by the
	// MDM. "managedDevice" means that the MDM's app is a device owner.
	// "managedProfile" means that the MDM's app is the profile owner (and
	// there is a separate personal profile which is not managed).
	// "containerApp" means that the MDM's app is managing the Android for
	// Work container app on the device.
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
}

func (s *Device) MarshalJSON() ([]byte, error) {
	type noMethod Device
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DeviceState) MarshalJSON() ([]byte, error) {
	type noMethod DeviceState
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *DevicesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DevicesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Enterprise: An enterprise resource represents a binding between an
// organisation and their MDM.
//
// To create an enterprise, an admin of the enterprise must first go
// through a Play for Work sign-up flow. At the end of this the admin
// will be presented with a token (a short opaque alphanumeric string).
// They must then present this to the MDM, who then supplies it to the
// enroll method. Until this is done the MDM will not have any access to
// the enterprise.
//
// After calling enroll the MDM should call setAccount to specify the
// service account that will be allowed to act on behalf of the
// enterprise, which will be required for access to the enterprise's
// data through this API. Only one call of setAccount is allowed for a
// given enterprise; the only way to change the account later is to
// unenroll the enterprise and enroll it again (obtaining a new
// token).
//
// The MDM can unenroll an enterprise in order to sever the binding
// between them. Re-enrolling an enterprise is possible, but requires a
// new token to be retrieved. Enterprises.unenroll requires the MDM's
// credentials (as enroll does), not the enterprise's.
// Enterprises.unenroll can only be used for enterprises that were
// previously enrolled with the enroll call. Any enterprises that were
// enrolled using the (deprecated) Enterprises.insert call must be
// unenrolled with Enterprises.delete and can then be re-enrolled using
// the Enterprises.enroll call.
//
// The ID for an enterprise is an opaque string. It is returned by
// insert and enroll and can also be retrieved if the enterprise's
// primary domain is known using the list method.
type Enterprise struct {
	// Id: The unique ID for the enterprise.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#enterprise".
	Kind string `json:"kind,omitempty"`

	// Name: The name of the enterprise, e.g. "Example Inc".
	Name string `json:"name,omitempty"`

	// PrimaryDomain: The enterprise's primary domain, e.g. "example.com".
	PrimaryDomain string `json:"primaryDomain,omitempty"`

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
}

func (s *Enterprise) MarshalJSON() ([]byte, error) {
	type noMethod Enterprise
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *EnterpriseAccount) MarshalJSON() ([]byte, error) {
	type noMethod EnterpriseAccount
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *EnterprisesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod EnterprisesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *EnterprisesSendTestPushNotificationResponse) MarshalJSON() ([]byte, error) {
	type noMethod EnterprisesSendTestPushNotificationResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Entitlement: The existence of an entitlement resource means that a
// user has the right to use a particular app on any of their devices.
// This might be because the app is free or because they have been
// allocated a license to the app from a group license purchased by the
// enterprise.
//
// It should always be true that a user has an app installed on one of
// their devices only if they have an entitlement to it. So if an
// entitlement is deleted, the app will be uninstalled from all devices.
// Similarly if the user installs an app (and is permitted to do so), or
// the MDM triggers an install of the app, an entitlement to that app is
// automatically created. If this is impossible - e.g. the enterprise
// has not purchased sufficient licenses - then installation
// fails.
//
// Note that entitlements are always user specific, not device specific;
// a user may have an entitlement even though they have not installed
// the app anywhere. Once they have an entitlement they can install the
// app on multiple devices.
//
// The API can be used to create an entitlement. If the app is a free
// app, a group license for that app is created. If it's a paid app,
// creating the entitlement consumes one license; it remains consumed
// until the entitlement is removed. Optionally an installation of the
// app on all the user's managed devices can be triggered at the time
// the entitlement is created. An entitlement cannot be created for an
// app if the app requires permissions that the enterprise has not yet
// accepted.
//
// Entitlements for paid apps that are due to purchases by the user on a
// non-managed profile will have "userPurchase" as entitlement reason;
// those entitlements cannot be removed via the API.
type Entitlement struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#entitlement".
	Kind string `json:"kind,omitempty"`

	// ProductId: The ID of the product that the entitlement is for, e.g.
	// "app:com.google.android.gm".
	ProductId string `json:"productId,omitempty"`

	// Reason: The reason for the entitlement, e.g. "free" for free apps.
	// This is temporary, it will be replaced by the acquisition kind field
	// of group licenses.
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
}

func (s *Entitlement) MarshalJSON() ([]byte, error) {
	type noMethod Entitlement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *EntitlementsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod EntitlementsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GroupLicense: A group license object indicates a product that an
// enterprise admin has approved for use in the enterprise. The product
// may be free or paid. For free products, a group license object is
// created in these cases: if the enterprise admin approves a product in
// Google Play, if the product is added to a collection, or if an
// entitlement for the product is created for a user via the API. For
// paid products, a group license object is only created as part of the
// first bulk purchase of that product in Google Play by the enterprise
// admin.
//
// The API can be used to query group licenses; the available
// information includes the total number of licenses purchased (for paid
// products) and the total number of licenses that have been
// provisioned, that is, the total number of user entitlements in
// existence for the product.
//
// Group license objects are never deleted. If, for example, a free app
// is added to a collection and then removed, the group license will
// remain, allowing the enterprise admin to keep track of any remaining
// entitlements. An enterprise admin may indicate they are no longer
// interested in the group license by marking it as unapproved in Google
// Play.
type GroupLicense struct {
	// AcquisitionKind: How this group license was acquired. "bulkPurchase"
	// means that this group license object was created because the
	// enterprise purchased licenses for this product; this is "free"
	// otherwise (for free products).
	AcquisitionKind string `json:"acquisitionKind,omitempty"`

	// Approval: Whether the product to which this group license relates is
	// currently approved by the enterprise, as either "approved" or
	// "unapproved". Products are approved when a group license is first
	// created, but this approval may be revoked by an enterprise admin via
	// Google Play. Unapproved products will not be visible to end users in
	// collections and new entitlements to them should not normally be
	// created.
	Approval string `json:"approval,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#groupLicense".
	Kind string `json:"kind,omitempty"`

	// NumProvisioned: The total number of provisioned licenses for this
	// product. Returned by read operations, but ignored in write
	// operations.
	NumProvisioned int64 `json:"numProvisioned,omitempty"`

	// NumPurchased: The number of purchased licenses (possibly in multiple
	// purchases). If this field is omitted then there is no limit on the
	// number of licenses that can be provisioned (e.g. if the acquisition
	// kind is "free").
	NumPurchased int64 `json:"numPurchased,omitempty"`

	// ProductId: The ID of the product that the license is for, e.g.
	// "app:com.google.android.gm".
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
}

func (s *GroupLicense) MarshalJSON() ([]byte, error) {
	type noMethod GroupLicense
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *GroupLicenseUsersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod GroupLicenseUsersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *GroupLicensesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod GroupLicensesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Install: The existence of an install resource indicates that an app
// is installed on a particular device (or that an install is
// pending).
//
// The API can be used to create an install resource using the update
// method. This triggers the actual install of the app on the device. If
// the user does not already have an entitlement for the app then an
// attempt is made to create one. If this fails (e.g. because the app is
// not free and there is no available license) then the creation of the
// install fails.
//
// The API can also be used to update an installed app. If the update
// method is used on an existing install then the app will be updated to
// the latest available version.
//
// Note that it is not possible to force the installation of a specific
// version of an app; the version code is read-only.
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

	// ProductId: The ID of the product that the install is for, e.g.
	// "app:com.google.android.gm".
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
}

func (s *Install) MarshalJSON() ([]byte, error) {
	type noMethod Install
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *InstallsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod InstallsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *LocalizedText) MarshalJSON() ([]byte, error) {
	type noMethod LocalizedText
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Permission: A permission represents some extra capability, to be
// granted to an Android app, which requires explicit consent. An
// enterprise admin must consent to these permissions on behalf of their
// users before an entitlement for the app can be created.
//
// The permissions collection is read-only. The information provided for
// each permission (localized name and description) is intended to be
// used in the MDM user interface when obtaining consent from the
// enterprise.
type Permission struct {
	// Description: A longer description of the permissions giving more
	// details of what it affects.
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
}

func (s *Permission) MarshalJSON() ([]byte, error) {
	type noMethod Permission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Product: A product represents an app in the Google Play Store that is
// available to at least some users in the enterprise. (Some apps are
// restricted to a single enterprise, and no information about them is
// made available outside that enterprise.)
//
// The information provided for each product (localized name, icon, link
// to the full Google Play details page) is intended to allow a basic
// representation of the product within an MDM user interface.
type Product struct {
	// AppVersion: App versions currently available for this product. The
	// returned list contains only public versions. Alpha and beta versions
	// are not included.
	AppVersion []*AppVersion `json:"appVersion,omitempty"`

	// AuthorName: The name of the author of the product (e.g. the app
	// developer).
	AuthorName string `json:"authorName,omitempty"`

	// DetailsUrl: A link to the (consumer) Google Play details page for the
	// product.
	DetailsUrl string `json:"detailsUrl,omitempty"`

	// DistributionChannel: How and to whom the package is made available.
	// The value publicGoogleHosted means that the package is available
	// through the Play Store and not restricted to a specific enterprise.
	// The value privateGoogleHosted means that the package is a private app
	// (restricted to an enterprise) but hosted by Google. The value
	// privateSelfHosted means that the package is a private app (restricted
	// to an enterprise) and is privately hosted.
	DistributionChannel string `json:"distributionChannel,omitempty"`

	// IconUrl: A link to an image that can be used as an icon for the
	// product.
	IconUrl string `json:"iconUrl,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#product".
	Kind string `json:"kind,omitempty"`

	// ProductId: A string of the form app:<package name>. For example,
	// app:com.google.android.gm represents the Gmail app.
	ProductId string `json:"productId,omitempty"`

	// RequiresContainerApp: Whether this app can only be installed on
	// devices using the Android for Work container app.
	RequiresContainerApp bool `json:"requiresContainerApp,omitempty"`

	// Title: The name of the product.
	Title string `json:"title,omitempty"`

	// WorkDetailsUrl: A link to the Google Play for Work details page for
	// the product, for use by an Enterprise administrator.
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
}

func (s *Product) MarshalJSON() ([]byte, error) {
	type noMethod Product
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductPermission) MarshalJSON() ([]byte, error) {
	type noMethod ProductPermission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductPermissions) MarshalJSON() ([]byte, error) {
	type noMethod ProductPermissions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ProductSet: A set of products.
type ProductSet struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#productSet".
	Kind string `json:"kind,omitempty"`

	// ProductId: The list of product IDs making up the set of products.
	ProductId []string `json:"productId,omitempty"`

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

func (s *ProductSet) MarshalJSON() ([]byte, error) {
	type noMethod ProductSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ProductsApproveRequest struct {
	// ApprovalUrlInfo: The approval URL that was shown to the user. Only
	// the permissions shown to the user with that URL will be accepted,
	// which may not be the product's entire set of permissions. For
	// example, the URL may only display new permissions from an update
	// after the product was approved, or not include new permissions if the
	// product was updated since the URL was generated.
	ApprovalUrlInfo *ApprovalUrlInfo `json:"approvalUrlInfo,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApprovalUrlInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ProductsApproveRequest) MarshalJSON() ([]byte, error) {
	type noMethod ProductsApproveRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ProductsGenerateApprovalUrlResponse) MarshalJSON() ([]byte, error) {
	type noMethod ProductsGenerateApprovalUrlResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// StoreCluster: Definition of a Google Play for Work store cluster, a
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
	// for the purpose of defining an ordering. Maximum length is 20
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
}

func (s *StoreCluster) MarshalJSON() ([]byte, error) {
	type noMethod StoreCluster
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// StoreLayout: General setting for the Google Play for Work store
// layout, currently only specifying the page to display the first time
// the store is opened.
type StoreLayout struct {
	// HomepageId: The ID of the store page to be used as the homepage. The
	// homepage will be used as the first page shown in the Google Play for
	// Work store.
	//
	// If there is no homepage set, an empty store is shown. The homepage
	// can be unset (by not specifying it) to empty the store.
	//
	// If there exists at least one page, this field must be set to the ID
	// of a valid page.
	HomepageId string `json:"homepageId,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#storeLayout".
	Kind string `json:"kind,omitempty"`

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
}

func (s *StoreLayout) MarshalJSON() ([]byte, error) {
	type noMethod StoreLayout
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *StoreLayoutClustersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod StoreLayoutClustersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *StoreLayoutPagesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod StoreLayoutPagesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// StorePage: Definition of a Google Play for Work store page, made of a
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
}

func (s *StorePage) MarshalJSON() ([]byte, error) {
	type noMethod StorePage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// User: A user resource represents an individual user within the
// enterprise's domain.
//
// Note that each user is associated with a Google account based on the
// user's corporate email address (which must be in one of the
// enterprise's domains). As part of installing an MDM app to manage a
// device the Google account must be provisioned to the device, and so
// the user resource must be created before that. This can be done using
// the Google Admin SDK Directory API.
//
// The ID for a user is an opaque string. It can be retrieved using the
// list method queried by the user's primary email address.
type User struct {
	// Id: The unique ID for the user.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "androidenterprise#user".
	Kind string `json:"kind,omitempty"`

	// PrimaryEmail: The user's primary email, e.g. "jsmith@example.com".
	// Will always be set for Google managed users and not set for EMM
	// managed users.
	PrimaryEmail string `json:"primaryEmail,omitempty"`

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
}

func (s *User) MarshalJSON() ([]byte, error) {
	type noMethod User
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UserToken: A UserToken is used by a user when setting up a managed
// device or profile with their work account on a device. When the user
// enters their email address and token (activation code) the
// appropriate MDM app can be automatically downloaded.
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
}

func (s *UserToken) MarshalJSON() ([]byte, error) {
	type noMethod UserToken
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *UsersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod UsersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "androidenterprise.collections.delete":

type CollectionsDeleteCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Delete: Deletes a collection.
func (r *CollectionsService) Delete(enterpriseId string, collectionId string) *CollectionsDeleteCall {
	c := &CollectionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionsDeleteCall) QuotaUser(quotaUser string) *CollectionsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionsDeleteCall) UserIP(userIP string) *CollectionsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionsDeleteCall) Fields(s ...googleapi.Field) *CollectionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionsDeleteCall) Context(ctx context.Context) *CollectionsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collections.delete" call.
func (c *CollectionsDeleteCall) Do() error {
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
	//   "description": "Deletes a collection.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.collections.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
	//       "location": "path",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collections.get":

type CollectionsGetCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the details of a collection.
func (r *CollectionsService) Get(enterpriseId string, collectionId string) *CollectionsGetCall {
	c := &CollectionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionsGetCall) QuotaUser(quotaUser string) *CollectionsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionsGetCall) UserIP(userIP string) *CollectionsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionsGetCall) Fields(s ...googleapi.Field) *CollectionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CollectionsGetCall) IfNoneMatch(entityTag string) *CollectionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionsGetCall) Context(ctx context.Context) *CollectionsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collections.get" call.
// Exactly one of *Collection or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Collection.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CollectionsGetCall) Do() (*Collection, error) {
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
	ret := &Collection{
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
	//   "description": "Retrieves the details of a collection.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.collections.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
	//       "location": "path",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}",
	//   "response": {
	//     "$ref": "Collection"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collections.insert":

type CollectionsInsertCall struct {
	s            *Service
	enterpriseId string
	collection   *Collection
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Insert: Creates a new collection.
func (r *CollectionsService) Insert(enterpriseId string, collection *Collection) *CollectionsInsertCall {
	c := &CollectionsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collection = collection
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionsInsertCall) QuotaUser(quotaUser string) *CollectionsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionsInsertCall) UserIP(userIP string) *CollectionsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionsInsertCall) Fields(s ...googleapi.Field) *CollectionsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionsInsertCall) Context(ctx context.Context) *CollectionsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.collection)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collections.insert" call.
// Exactly one of *Collection or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Collection.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CollectionsInsertCall) Do() (*Collection, error) {
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
	ret := &Collection{
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
	//   "description": "Creates a new collection.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.collections.insert",
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
	//   "path": "enterprises/{enterpriseId}/collections",
	//   "request": {
	//     "$ref": "Collection"
	//   },
	//   "response": {
	//     "$ref": "Collection"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collections.list":

type CollectionsListCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves the IDs of all the collections for an enterprise.
func (r *CollectionsService) List(enterpriseId string) *CollectionsListCall {
	c := &CollectionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionsListCall) QuotaUser(quotaUser string) *CollectionsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionsListCall) UserIP(userIP string) *CollectionsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionsListCall) Fields(s ...googleapi.Field) *CollectionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CollectionsListCall) IfNoneMatch(entityTag string) *CollectionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionsListCall) Context(ctx context.Context) *CollectionsListCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collections.list" call.
// Exactly one of *CollectionsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *CollectionsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CollectionsListCall) Do() (*CollectionsListResponse, error) {
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
	ret := &CollectionsListResponse{
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
	//   "description": "Retrieves the IDs of all the collections for an enterprise.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.collections.list",
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
	//   "path": "enterprises/{enterpriseId}/collections",
	//   "response": {
	//     "$ref": "CollectionsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collections.patch":

type CollectionsPatchCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	collection   *Collection
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Patch: Updates a collection. This method supports patch semantics.
func (r *CollectionsService) Patch(enterpriseId string, collectionId string, collection *Collection) *CollectionsPatchCall {
	c := &CollectionsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	c.collection = collection
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionsPatchCall) QuotaUser(quotaUser string) *CollectionsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionsPatchCall) UserIP(userIP string) *CollectionsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionsPatchCall) Fields(s ...googleapi.Field) *CollectionsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionsPatchCall) Context(ctx context.Context) *CollectionsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.collection)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collections.patch" call.
// Exactly one of *Collection or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Collection.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CollectionsPatchCall) Do() (*Collection, error) {
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
	ret := &Collection{
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
	//   "description": "Updates a collection. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.collections.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
	//       "location": "path",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}",
	//   "request": {
	//     "$ref": "Collection"
	//   },
	//   "response": {
	//     "$ref": "Collection"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collections.update":

type CollectionsUpdateCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	collection   *Collection
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Update: Updates a collection.
func (r *CollectionsService) Update(enterpriseId string, collectionId string, collection *Collection) *CollectionsUpdateCall {
	c := &CollectionsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	c.collection = collection
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionsUpdateCall) QuotaUser(quotaUser string) *CollectionsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionsUpdateCall) UserIP(userIP string) *CollectionsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionsUpdateCall) Fields(s ...googleapi.Field) *CollectionsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionsUpdateCall) Context(ctx context.Context) *CollectionsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.collection)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collections.update" call.
// Exactly one of *Collection or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Collection.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CollectionsUpdateCall) Do() (*Collection, error) {
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
	ret := &Collection{
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
	//   "description": "Updates a collection.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.collections.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
	//       "location": "path",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}",
	//   "request": {
	//     "$ref": "Collection"
	//   },
	//   "response": {
	//     "$ref": "Collection"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collectionviewers.delete":

type CollectionviewersDeleteCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	userId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Delete: Removes the user from the list of those specifically allowed
// to see the collection. If the collection's visibility is set to
// viewersOnly then only such users will see the collection.
func (r *CollectionviewersService) Delete(enterpriseId string, collectionId string, userId string) *CollectionviewersDeleteCall {
	c := &CollectionviewersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionviewersDeleteCall) QuotaUser(quotaUser string) *CollectionviewersDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionviewersDeleteCall) UserIP(userIP string) *CollectionviewersDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionviewersDeleteCall) Fields(s ...googleapi.Field) *CollectionviewersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionviewersDeleteCall) Context(ctx context.Context) *CollectionviewersDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionviewersDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collectionviewers.delete" call.
func (c *CollectionviewersDeleteCall) Do() error {
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
	//   "description": "Removes the user from the list of those specifically allowed to see the collection. If the collection's visibility is set to viewersOnly then only such users will see the collection.",
	//   "httpMethod": "DELETE",
	//   "id": "androidenterprise.collectionviewers.delete",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collectionviewers.get":

type CollectionviewersGetCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the ID of the user if they have been specifically
// allowed to see the collection. If the collection's visibility is set
// to viewersOnly then only these users will see the collection.
func (r *CollectionviewersService) Get(enterpriseId string, collectionId string, userId string) *CollectionviewersGetCall {
	c := &CollectionviewersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionviewersGetCall) QuotaUser(quotaUser string) *CollectionviewersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionviewersGetCall) UserIP(userIP string) *CollectionviewersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionviewersGetCall) Fields(s ...googleapi.Field) *CollectionviewersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CollectionviewersGetCall) IfNoneMatch(entityTag string) *CollectionviewersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionviewersGetCall) Context(ctx context.Context) *CollectionviewersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionviewersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collectionviewers.get" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *CollectionviewersGetCall) Do() (*User, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the ID of the user if they have been specifically allowed to see the collection. If the collection's visibility is set to viewersOnly then only these users will see the collection.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.collectionviewers.get",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}",
	//   "response": {
	//     "$ref": "User"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collectionviewers.list":

type CollectionviewersListCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves the IDs of the users who have been specifically
// allowed to see the collection. If the collection's visibility is set
// to viewersOnly then only these users will see the collection.
func (r *CollectionviewersService) List(enterpriseId string, collectionId string) *CollectionviewersListCall {
	c := &CollectionviewersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionviewersListCall) QuotaUser(quotaUser string) *CollectionviewersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionviewersListCall) UserIP(userIP string) *CollectionviewersListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionviewersListCall) Fields(s ...googleapi.Field) *CollectionviewersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CollectionviewersListCall) IfNoneMatch(entityTag string) *CollectionviewersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionviewersListCall) Context(ctx context.Context) *CollectionviewersListCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionviewersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collectionviewers.list" call.
// Exactly one of *CollectionViewersListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *CollectionViewersListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CollectionviewersListCall) Do() (*CollectionViewersListResponse, error) {
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
	ret := &CollectionViewersListResponse{
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
	//   "description": "Retrieves the IDs of the users who have been specifically allowed to see the collection. If the collection's visibility is set to viewersOnly then only these users will see the collection.",
	//   "httpMethod": "GET",
	//   "id": "androidenterprise.collectionviewers.list",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
	//       "location": "path",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}/users",
	//   "response": {
	//     "$ref": "CollectionViewersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidenterprise"
	//   ]
	// }

}

// method id "androidenterprise.collectionviewers.patch":

type CollectionviewersPatchCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	userId       string
	user         *User
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Patch: Adds the user to the list of those specifically allowed to see
// the collection. If the collection's visibility is set to viewersOnly
// then only such users will see the collection. This method supports
// patch semantics.
func (r *CollectionviewersService) Patch(enterpriseId string, collectionId string, userId string, user *User) *CollectionviewersPatchCall {
	c := &CollectionviewersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	c.userId = userId
	c.user = user
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionviewersPatchCall) QuotaUser(quotaUser string) *CollectionviewersPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionviewersPatchCall) UserIP(userIP string) *CollectionviewersPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionviewersPatchCall) Fields(s ...googleapi.Field) *CollectionviewersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionviewersPatchCall) Context(ctx context.Context) *CollectionviewersPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionviewersPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
		"userId":       c.userId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collectionviewers.patch" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *CollectionviewersPatchCall) Do() (*User, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds the user to the list of those specifically allowed to see the collection. If the collection's visibility is set to viewersOnly then only such users will see the collection. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "androidenterprise.collectionviewers.patch",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}",
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

// method id "androidenterprise.collectionviewers.update":

type CollectionviewersUpdateCall struct {
	s            *Service
	enterpriseId string
	collectionId string
	userId       string
	user         *User
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Update: Adds the user to the list of those specifically allowed to
// see the collection. If the collection's visibility is set to
// viewersOnly then only such users will see the collection.
func (r *CollectionviewersService) Update(enterpriseId string, collectionId string, userId string, user *User) *CollectionviewersUpdateCall {
	c := &CollectionviewersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.collectionId = collectionId
	c.userId = userId
	c.user = user
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *CollectionviewersUpdateCall) QuotaUser(quotaUser string) *CollectionviewersUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *CollectionviewersUpdateCall) UserIP(userIP string) *CollectionviewersUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CollectionviewersUpdateCall) Fields(s ...googleapi.Field) *CollectionviewersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CollectionviewersUpdateCall) Context(ctx context.Context) *CollectionviewersUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *CollectionviewersUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"collectionId": c.collectionId,
		"userId":       c.userId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.collectionviewers.update" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *CollectionviewersUpdateCall) Do() (*User, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds the user to the list of those specifically allowed to see the collection. If the collection's visibility is set to viewersOnly then only such users will see the collection.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.collectionviewers.update",
	//   "parameterOrder": [
	//     "enterpriseId",
	//     "collectionId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "collectionId": {
	//       "description": "The ID of the collection.",
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
	//   "path": "enterprises/{enterpriseId}/collections/{collectionId}/users/{userId}",
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

// method id "androidenterprise.devices.get":

type DevicesGetCall struct {
	s            *Service
	enterpriseId string
	userId       string
	deviceId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the details of a device.
func (r *DevicesService) Get(enterpriseId string, userId string, deviceId string) *DevicesGetCall {
	c := &DevicesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DevicesGetCall) QuotaUser(quotaUser string) *DevicesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DevicesGetCall) UserIP(userIP string) *DevicesGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DevicesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.devices.get" call.
// Exactly one of *Device or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Device.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DevicesGetCall) Do() (*Device, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// GetState: Retrieves whether a device is enabled or disabled for
// access by the user to Google services. The device state takes effect
// only if enforcing EMM policies on Android devices is enabled in the
// Google Admin Console. Otherwise, the device state is ignored and all
// devices are allowed access to Google services.
func (r *DevicesService) GetState(enterpriseId string, userId string, deviceId string) *DevicesGetStateCall {
	c := &DevicesGetStateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DevicesGetStateCall) QuotaUser(quotaUser string) *DevicesGetStateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DevicesGetStateCall) UserIP(userIP string) *DevicesGetStateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DevicesGetStateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/state")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.devices.getState" call.
// Exactly one of *DeviceState or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DeviceState.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DevicesGetStateCall) Do() (*DeviceState, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves whether a device is enabled or disabled for access by the user to Google services. The device state takes effect only if enforcing EMM policies on Android devices is enabled in the Google Admin Console. Otherwise, the device state is ignored and all devices are allowed access to Google services.",
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
}

// List: Retrieves the IDs of all of a user's devices.
func (r *DevicesService) List(enterpriseId string, userId string) *DevicesListCall {
	c := &DevicesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DevicesListCall) QuotaUser(quotaUser string) *DevicesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DevicesListCall) UserIP(userIP string) *DevicesListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DevicesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.devices.list" call.
// Exactly one of *DevicesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DevicesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DevicesListCall) Do() (*DevicesListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// SetState: Sets whether a device is enabled or disabled for access by
// the user to Google services. The device state takes effect only if
// enforcing EMM policies on Android devices is enabled in the Google
// Admin Console. Otherwise, the device state is ignored and all devices
// are allowed access to Google services.
func (r *DevicesService) SetState(enterpriseId string, userId string, deviceId string, devicestate *DeviceState) *DevicesSetStateCall {
	c := &DevicesSetStateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.deviceId = deviceId
	c.devicestate = devicestate
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DevicesSetStateCall) QuotaUser(quotaUser string) *DevicesSetStateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DevicesSetStateCall) UserIP(userIP string) *DevicesSetStateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *DevicesSetStateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.devicestate)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/state")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.devices.setState" call.
// Exactly one of *DeviceState or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DeviceState.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DevicesSetStateCall) Do() (*DeviceState, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets whether a device is enabled or disabled for access by the user to Google services. The device state takes effect only if enforcing EMM policies on Android devices is enabled in the Google Admin Console. Otherwise, the device state is ignored and all devices are allowed access to Google services.",
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

// method id "androidenterprise.enterprises.delete":

type EnterprisesDeleteCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Delete: Deletes the binding between the MDM and enterprise. This is
// now deprecated; use this to unenroll customers that were previously
// enrolled with the 'insert' call, then enroll them again with the
// 'enroll' call.
func (r *EnterprisesService) Delete(enterpriseId string) *EnterprisesDeleteCall {
	c := &EnterprisesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesDeleteCall) QuotaUser(quotaUser string) *EnterprisesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesDeleteCall) UserIP(userIP string) *EnterprisesDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.enterprises.delete" call.
func (c *EnterprisesDeleteCall) Do() error {
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
	//   "description": "Deletes the binding between the MDM and enterprise. This is now deprecated; use this to unenroll customers that were previously enrolled with the 'insert' call, then enroll them again with the 'enroll' call.",
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
}

// Enroll: Enrolls an enterprise with the calling MDM.
func (r *EnterprisesService) Enroll(token string, enterprise *Enterprise) *EnterprisesEnrollCall {
	c := &EnterprisesEnrollCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("token", token)
	c.enterprise = enterprise
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesEnrollCall) QuotaUser(quotaUser string) *EnterprisesEnrollCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesEnrollCall) UserIP(userIP string) *EnterprisesEnrollCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesEnrollCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enterprise)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/enroll")
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

// Do executes the "androidenterprise.enterprises.enroll" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesEnrollCall) Do() (*Enterprise, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Enrolls an enterprise with the calling MDM.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.enroll",
	//   "parameterOrder": [
	//     "token"
	//   ],
	//   "parameters": {
	//     "token": {
	//       "description": "The token provided by the enterprise to register the MDM.",
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

// method id "androidenterprise.enterprises.get":

type EnterprisesGetCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the name and domain of an enterprise.
func (r *EnterprisesService) Get(enterpriseId string) *EnterprisesGetCall {
	c := &EnterprisesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesGetCall) QuotaUser(quotaUser string) *EnterprisesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesGetCall) UserIP(userIP string) *EnterprisesGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.enterprises.get" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesGetCall) Do() (*Enterprise, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// method id "androidenterprise.enterprises.getStoreLayout":

type EnterprisesGetStoreLayoutCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// GetStoreLayout: Returns the store layout resource.
func (r *EnterprisesService) GetStoreLayout(enterpriseId string) *EnterprisesGetStoreLayoutCall {
	c := &EnterprisesGetStoreLayoutCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesGetStoreLayoutCall) QuotaUser(quotaUser string) *EnterprisesGetStoreLayoutCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesGetStoreLayoutCall) UserIP(userIP string) *EnterprisesGetStoreLayoutCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesGetStoreLayoutCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.enterprises.getStoreLayout" call.
// Exactly one of *StoreLayout or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreLayout.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesGetStoreLayoutCall) Do() (*StoreLayout, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the store layout resource.",
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
}

// Insert: Establishes the binding between the MDM and an enterprise.
// This is now deprecated; use enroll instead.
func (r *EnterprisesService) Insert(token string, enterprise *Enterprise) *EnterprisesInsertCall {
	c := &EnterprisesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("token", token)
	c.enterprise = enterprise
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesInsertCall) QuotaUser(quotaUser string) *EnterprisesInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesInsertCall) UserIP(userIP string) *EnterprisesInsertCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enterprise)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises")
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

// Do executes the "androidenterprise.enterprises.insert" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesInsertCall) Do() (*Enterprise, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Establishes the binding between the MDM and an enterprise. This is now deprecated; use enroll instead.",
	//   "httpMethod": "POST",
	//   "id": "androidenterprise.enterprises.insert",
	//   "parameterOrder": [
	//     "token"
	//   ],
	//   "parameters": {
	//     "token": {
	//       "description": "The token provided by the enterprise to register the MDM.",
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
}

// List: Looks up an enterprise by domain name.
func (r *EnterprisesService) List(domain string) *EnterprisesListCall {
	c := &EnterprisesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("domain", domain)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesListCall) QuotaUser(quotaUser string) *EnterprisesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesListCall) UserIP(userIP string) *EnterprisesListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises")
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

// Do executes the "androidenterprise.enterprises.list" call.
// Exactly one of *EnterprisesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *EnterprisesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesListCall) Do() (*EnterprisesListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Looks up an enterprise by domain name.",
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

// method id "androidenterprise.enterprises.sendTestPushNotification":

type EnterprisesSendTestPushNotificationCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// SendTestPushNotification: Sends a test push notification to validate
// the MDM integration with the Google Cloud Pub/Sub service for this
// enterprise.
func (r *EnterprisesService) SendTestPushNotification(enterpriseId string) *EnterprisesSendTestPushNotificationCall {
	c := &EnterprisesSendTestPushNotificationCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesSendTestPushNotificationCall) QuotaUser(quotaUser string) *EnterprisesSendTestPushNotificationCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesSendTestPushNotificationCall) UserIP(userIP string) *EnterprisesSendTestPushNotificationCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesSendTestPushNotificationCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/sendTestPushNotification")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
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
func (c *EnterprisesSendTestPushNotificationCall) Do() (*EnterprisesSendTestPushNotificationResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sends a test push notification to validate the MDM integration with the Google Cloud Pub/Sub service for this enterprise.",
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
}

// SetAccount: Set the account that will be used to authenticate to the
// API as the enterprise.
func (r *EnterprisesService) SetAccount(enterpriseId string, enterpriseaccount *EnterpriseAccount) *EnterprisesSetAccountCall {
	c := &EnterprisesSetAccountCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.enterpriseaccount = enterpriseaccount
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesSetAccountCall) QuotaUser(quotaUser string) *EnterprisesSetAccountCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesSetAccountCall) UserIP(userIP string) *EnterprisesSetAccountCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesSetAccountCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enterpriseaccount)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/account")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.enterprises.setAccount" call.
// Exactly one of *EnterpriseAccount or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *EnterpriseAccount.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesSetAccountCall) Do() (*EnterpriseAccount, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Set the account that will be used to authenticate to the API as the enterprise.",
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

// method id "androidenterprise.enterprises.setStoreLayout":

type EnterprisesSetStoreLayoutCall struct {
	s            *Service
	enterpriseId string
	storelayout  *StoreLayout
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// SetStoreLayout: Sets the store layout resource.
func (r *EnterprisesService) SetStoreLayout(enterpriseId string, storelayout *StoreLayout) *EnterprisesSetStoreLayoutCall {
	c := &EnterprisesSetStoreLayoutCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.storelayout = storelayout
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesSetStoreLayoutCall) QuotaUser(quotaUser string) *EnterprisesSetStoreLayoutCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesSetStoreLayoutCall) UserIP(userIP string) *EnterprisesSetStoreLayoutCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesSetStoreLayoutCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storelayout)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.enterprises.setStoreLayout" call.
// Exactly one of *StoreLayout or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreLayout.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesSetStoreLayoutCall) Do() (*StoreLayout, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the store layout resource.",
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
}

// Unenroll: Unenrolls an enterprise from the calling MDM.
func (r *EnterprisesService) Unenroll(enterpriseId string) *EnterprisesUnenrollCall {
	c := &EnterprisesUnenrollCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EnterprisesUnenrollCall) QuotaUser(quotaUser string) *EnterprisesUnenrollCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EnterprisesUnenrollCall) UserIP(userIP string) *EnterprisesUnenrollCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EnterprisesUnenrollCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/unenroll")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.enterprises.unenroll" call.
func (c *EnterprisesUnenrollCall) Do() error {
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
	//   "description": "Unenrolls an enterprise from the calling MDM.",
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
}

// Delete: Removes an entitlement to an app for a user and uninstalls
// it.
func (r *EntitlementsService) Delete(enterpriseId string, userId string, entitlementId string) *EntitlementsDeleteCall {
	c := &EntitlementsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.entitlementId = entitlementId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EntitlementsDeleteCall) QuotaUser(quotaUser string) *EntitlementsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EntitlementsDeleteCall) UserIP(userIP string) *EntitlementsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EntitlementsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.entitlements.delete" call.
func (c *EntitlementsDeleteCall) Do() error {
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
	//   "description": "Removes an entitlement to an app for a user and uninstalls it.",
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
	//       "description": "The ID of the entitlement, e.g. \"app:com.google.android.gm\".",
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
}

// Get: Retrieves details of an entitlement.
func (r *EntitlementsService) Get(enterpriseId string, userId string, entitlementId string) *EntitlementsGetCall {
	c := &EntitlementsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.entitlementId = entitlementId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EntitlementsGetCall) QuotaUser(quotaUser string) *EntitlementsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EntitlementsGetCall) UserIP(userIP string) *EntitlementsGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EntitlementsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.entitlements.get" call.
// Exactly one of *Entitlement or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Entitlement.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EntitlementsGetCall) Do() (*Entitlement, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "The ID of the entitlement, e.g. \"app:com.google.android.gm\".",
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
}

// List: List of all entitlements for the specified user. Only the ID is
// set.
func (r *EntitlementsService) List(enterpriseId string, userId string) *EntitlementsListCall {
	c := &EntitlementsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EntitlementsListCall) QuotaUser(quotaUser string) *EntitlementsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EntitlementsListCall) UserIP(userIP string) *EntitlementsListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EntitlementsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.entitlements.list" call.
// Exactly one of *EntitlementsListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *EntitlementsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EntitlementsListCall) Do() (*EntitlementsListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List of all entitlements for the specified user. Only the ID is set.",
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EntitlementsPatchCall) QuotaUser(quotaUser string) *EntitlementsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EntitlementsPatchCall) UserIP(userIP string) *EntitlementsPatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EntitlementsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entitlement)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.entitlements.patch" call.
// Exactly one of *Entitlement or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Entitlement.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EntitlementsPatchCall) Do() (*Entitlement, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "The ID of the entitlement, e.g. \"app:com.google.android.gm\".",
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *EntitlementsUpdateCall) QuotaUser(quotaUser string) *EntitlementsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *EntitlementsUpdateCall) UserIP(userIP string) *EntitlementsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *EntitlementsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entitlement)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/entitlements/{entitlementId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":  c.enterpriseId,
		"userId":        c.userId,
		"entitlementId": c.entitlementId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.entitlements.update" call.
// Exactly one of *Entitlement or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Entitlement.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EntitlementsUpdateCall) Do() (*Entitlement, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "The ID of the entitlement, e.g. \"app:com.google.android.gm\".",
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
}

// Get: Retrieves details of an enterprise's group license for a
// product.
func (r *GrouplicensesService) Get(enterpriseId string, groupLicenseId string) *GrouplicensesGetCall {
	c := &GrouplicensesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.groupLicenseId = groupLicenseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GrouplicensesGetCall) QuotaUser(quotaUser string) *GrouplicensesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GrouplicensesGetCall) UserIP(userIP string) *GrouplicensesGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *GrouplicensesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/groupLicenses/{groupLicenseId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":   c.enterpriseId,
		"groupLicenseId": c.groupLicenseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.grouplicenses.get" call.
// Exactly one of *GroupLicense or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *GroupLicense.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GrouplicensesGetCall) Do() (*GroupLicense, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// List: Retrieves IDs of all products for which the enterprise has a
// group license.
func (r *GrouplicensesService) List(enterpriseId string) *GrouplicensesListCall {
	c := &GrouplicensesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GrouplicensesListCall) QuotaUser(quotaUser string) *GrouplicensesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GrouplicensesListCall) UserIP(userIP string) *GrouplicensesListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *GrouplicensesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/groupLicenses")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.grouplicenses.list" call.
// Exactly one of *GroupLicensesListResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GroupLicensesListResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GrouplicensesListCall) Do() (*GroupLicensesListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// List: Retrieves the IDs of the users who have been granted
// entitlements under the license.
func (r *GrouplicenseusersService) List(enterpriseId string, groupLicenseId string) *GrouplicenseusersListCall {
	c := &GrouplicenseusersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.groupLicenseId = groupLicenseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GrouplicenseusersListCall) QuotaUser(quotaUser string) *GrouplicenseusersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GrouplicenseusersListCall) UserIP(userIP string) *GrouplicenseusersListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *GrouplicenseusersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/groupLicenses/{groupLicenseId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId":   c.enterpriseId,
		"groupLicenseId": c.groupLicenseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.grouplicenseusers.list" call.
// Exactly one of *GroupLicenseUsersListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *GroupLicenseUsersListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GrouplicenseusersListCall) Do() (*GroupLicenseUsersListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InstallsDeleteCall) QuotaUser(quotaUser string) *InstallsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InstallsDeleteCall) UserIP(userIP string) *InstallsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *InstallsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.installs.delete" call.
func (c *InstallsDeleteCall) Do() error {
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InstallsGetCall) QuotaUser(quotaUser string) *InstallsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InstallsGetCall) UserIP(userIP string) *InstallsGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *InstallsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.installs.get" call.
// Exactly one of *Install or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Install.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InstallsGetCall) Do() (*Install, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InstallsListCall) QuotaUser(quotaUser string) *InstallsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InstallsListCall) UserIP(userIP string) *InstallsListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *InstallsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.installs.list" call.
// Exactly one of *InstallsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *InstallsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InstallsListCall) Do() (*InstallsListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// Patch: Requests to install the latest version of an app to a device.
// If the app is already installed then it is updated to the latest
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InstallsPatchCall) QuotaUser(quotaUser string) *InstallsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InstallsPatchCall) UserIP(userIP string) *InstallsPatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *InstallsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.install)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.installs.patch" call.
// Exactly one of *Install or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Install.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InstallsPatchCall) Do() (*Install, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Requests to install the latest version of an app to a device. If the app is already installed then it is updated to the latest version if necessary. This method supports patch semantics.",
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
}

// Update: Requests to install the latest version of an app to a device.
// If the app is already installed then it is updated to the latest
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *InstallsUpdateCall) QuotaUser(quotaUser string) *InstallsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *InstallsUpdateCall) UserIP(userIP string) *InstallsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *InstallsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.install)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/devices/{deviceId}/installs/{installId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
		"deviceId":     c.deviceId,
		"installId":    c.installId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.installs.update" call.
// Exactly one of *Install or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Install.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InstallsUpdateCall) Do() (*Install, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Requests to install the latest version of an app to a device. If the app is already installed then it is updated to the latest version if necessary.",
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

// method id "androidenterprise.permissions.get":

type PermissionsGetCall struct {
	s            *Service
	permissionId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *PermissionsGetCall) QuotaUser(quotaUser string) *PermissionsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *PermissionsGetCall) UserIP(userIP string) *PermissionsGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *PermissionsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "permissions/{permissionId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"permissionId": c.permissionId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.permissions.get" call.
// Exactly one of *Permission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Permission.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PermissionsGetCall) Do() (*Permission, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// Approve: Approves the specified product (and the relevant app
// permissions, if any).
func (r *ProductsService) Approve(enterpriseId string, productId string, productsapproverequest *ProductsApproveRequest) *ProductsApproveCall {
	c := &ProductsApproveCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	c.productsapproverequest = productsapproverequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsApproveCall) QuotaUser(quotaUser string) *ProductsApproveCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsApproveCall) UserIP(userIP string) *ProductsApproveCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsApproveCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productsapproverequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/approve")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.products.approve" call.
func (c *ProductsApproveCall) Do() error {
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
	//   "description": "Approves the specified product (and the relevant app permissions, if any).",
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsGenerateApprovalUrlCall) QuotaUser(quotaUser string) *ProductsGenerateApprovalUrlCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsGenerateApprovalUrlCall) UserIP(userIP string) *ProductsGenerateApprovalUrlCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsGenerateApprovalUrlCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/generateApprovalUrl")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.products.generateApprovalUrl" call.
// Exactly one of *ProductsGenerateApprovalUrlResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ProductsGenerateApprovalUrlResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProductsGenerateApprovalUrlCall) Do() (*ProductsGenerateApprovalUrlResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsGetCall) QuotaUser(quotaUser string) *ProductsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsGetCall) UserIP(userIP string) *ProductsGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.products.get" call.
// Exactly one of *Product or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Product.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProductsGetCall) Do() (*Product, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// GetAppRestrictionsSchema: Retrieves the schema defining app
// restrictions configurable for this product. All products have a
// schema, but this may be empty if no app restrictions are defined.
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsGetAppRestrictionsSchemaCall) QuotaUser(quotaUser string) *ProductsGetAppRestrictionsSchemaCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsGetAppRestrictionsSchemaCall) UserIP(userIP string) *ProductsGetAppRestrictionsSchemaCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsGetAppRestrictionsSchemaCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/appRestrictionsSchema")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.products.getAppRestrictionsSchema" call.
// Exactly one of *AppRestrictionsSchema or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AppRestrictionsSchema.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsGetAppRestrictionsSchemaCall) Do() (*AppRestrictionsSchema, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the schema defining app restrictions configurable for this product. All products have a schema, but this may be empty if no app restrictions are defined.",
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
}

// GetPermissions: Retrieves the Android app permissions required by
// this app.
func (r *ProductsService) GetPermissions(enterpriseId string, productId string) *ProductsGetPermissionsCall {
	c := &ProductsGetPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsGetPermissionsCall) QuotaUser(quotaUser string) *ProductsGetPermissionsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsGetPermissionsCall) UserIP(userIP string) *ProductsGetPermissionsCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ProductsGetPermissionsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/permissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.products.getPermissions" call.
// Exactly one of *ProductPermissions or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ProductPermissions.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsGetPermissionsCall) Do() (*ProductPermissions, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// method id "androidenterprise.products.updatePermissions":

type ProductsUpdatePermissionsCall struct {
	s                  *Service
	enterpriseId       string
	productId          string
	productpermissions *ProductPermissions
	urlParams_         gensupport.URLParams
	ctx_               context.Context
}

// UpdatePermissions: Updates the set of Android app permissions for
// this app that have been accepted by the enterprise.
func (r *ProductsService) UpdatePermissions(enterpriseId string, productId string, productpermissions *ProductPermissions) *ProductsUpdatePermissionsCall {
	c := &ProductsUpdatePermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.productId = productId
	c.productpermissions = productpermissions
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ProductsUpdatePermissionsCall) QuotaUser(quotaUser string) *ProductsUpdatePermissionsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ProductsUpdatePermissionsCall) UserIP(userIP string) *ProductsUpdatePermissionsCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProductsUpdatePermissionsCall) Fields(s ...googleapi.Field) *ProductsUpdatePermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProductsUpdatePermissionsCall) Context(ctx context.Context) *ProductsUpdatePermissionsCall {
	c.ctx_ = ctx
	return c
}

func (c *ProductsUpdatePermissionsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productpermissions)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/products/{productId}/permissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"productId":    c.productId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.products.updatePermissions" call.
// Exactly one of *ProductPermissions or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ProductPermissions.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProductsUpdatePermissionsCall) Do() (*ProductPermissions, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the set of Android app permissions for this app that have been accepted by the enterprise.",
	//   "httpMethod": "PUT",
	//   "id": "androidenterprise.products.updatePermissions",
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
	//   "request": {
	//     "$ref": "ProductPermissions"
	//   },
	//   "response": {
	//     "$ref": "ProductPermissions"
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
}

// Delete: Deletes a cluster.
func (r *StorelayoutclustersService) Delete(enterpriseId string, pageId string, clusterId string) *StorelayoutclustersDeleteCall {
	c := &StorelayoutclustersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.clusterId = clusterId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutclustersDeleteCall) QuotaUser(quotaUser string) *StorelayoutclustersDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutclustersDeleteCall) UserIP(userIP string) *StorelayoutclustersDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutclustersDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutclusters.delete" call.
func (c *StorelayoutclustersDeleteCall) Do() error {
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
}

// Get: Retrieves details of a cluster.
func (r *StorelayoutclustersService) Get(enterpriseId string, pageId string, clusterId string) *StorelayoutclustersGetCall {
	c := &StorelayoutclustersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.clusterId = clusterId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutclustersGetCall) QuotaUser(quotaUser string) *StorelayoutclustersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutclustersGetCall) UserIP(userIP string) *StorelayoutclustersGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutclustersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutclusters.get" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersGetCall) Do() (*StoreCluster, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// Insert: Inserts a new cluster in a page.
func (r *StorelayoutclustersService) Insert(enterpriseId string, pageId string, storecluster *StoreCluster) *StorelayoutclustersInsertCall {
	c := &StorelayoutclustersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.storecluster = storecluster
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutclustersInsertCall) QuotaUser(quotaUser string) *StorelayoutclustersInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutclustersInsertCall) UserIP(userIP string) *StorelayoutclustersInsertCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutclustersInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storecluster)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutclusters.insert" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersInsertCall) Do() (*StoreCluster, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// List: Retrieves the details of all clusters on the specified page.
func (r *StorelayoutclustersService) List(enterpriseId string, pageId string) *StorelayoutclustersListCall {
	c := &StorelayoutclustersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutclustersListCall) QuotaUser(quotaUser string) *StorelayoutclustersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutclustersListCall) UserIP(userIP string) *StorelayoutclustersListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutclustersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutclusters.list" call.
// Exactly one of *StoreLayoutClustersListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *StoreLayoutClustersListResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *StorelayoutclustersListCall) Do() (*StoreLayoutClustersListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutclustersPatchCall) QuotaUser(quotaUser string) *StorelayoutclustersPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutclustersPatchCall) UserIP(userIP string) *StorelayoutclustersPatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutclustersPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storecluster)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutclusters.patch" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersPatchCall) Do() (*StoreCluster, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutclustersUpdateCall) QuotaUser(quotaUser string) *StorelayoutclustersUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutclustersUpdateCall) UserIP(userIP string) *StorelayoutclustersUpdateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutclustersUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storecluster)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
		"clusterId":    c.clusterId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutclusters.update" call.
// Exactly one of *StoreCluster or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StoreCluster.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutclustersUpdateCall) Do() (*StoreCluster, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// Delete: Deletes a store page.
func (r *StorelayoutpagesService) Delete(enterpriseId string, pageId string) *StorelayoutpagesDeleteCall {
	c := &StorelayoutpagesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutpagesDeleteCall) QuotaUser(quotaUser string) *StorelayoutpagesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutpagesDeleteCall) UserIP(userIP string) *StorelayoutpagesDeleteCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutpagesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutpages.delete" call.
func (c *StorelayoutpagesDeleteCall) Do() error {
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
}

// Get: Retrieves details of a store page.
func (r *StorelayoutpagesService) Get(enterpriseId string, pageId string) *StorelayoutpagesGetCall {
	c := &StorelayoutpagesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutpagesGetCall) QuotaUser(quotaUser string) *StorelayoutpagesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutpagesGetCall) UserIP(userIP string) *StorelayoutpagesGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutpagesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutpages.get" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesGetCall) Do() (*StorePage, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// Insert: Inserts a new store page.
func (r *StorelayoutpagesService) Insert(enterpriseId string, storepage *StorePage) *StorelayoutpagesInsertCall {
	c := &StorelayoutpagesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.storepage = storepage
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutpagesInsertCall) QuotaUser(quotaUser string) *StorelayoutpagesInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutpagesInsertCall) UserIP(userIP string) *StorelayoutpagesInsertCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutpagesInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storepage)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutpages.insert" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesInsertCall) Do() (*StorePage, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// List: Retrieves the details of all pages in the store.
func (r *StorelayoutpagesService) List(enterpriseId string) *StorelayoutpagesListCall {
	c := &StorelayoutpagesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutpagesListCall) QuotaUser(quotaUser string) *StorelayoutpagesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutpagesListCall) UserIP(userIP string) *StorelayoutpagesListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutpagesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutpages.list" call.
// Exactly one of *StoreLayoutPagesListResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *StoreLayoutPagesListResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *StorelayoutpagesListCall) Do() (*StoreLayoutPagesListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutpagesPatchCall) QuotaUser(quotaUser string) *StorelayoutpagesPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutpagesPatchCall) UserIP(userIP string) *StorelayoutpagesPatchCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutpagesPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storepage)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutpages.patch" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesPatchCall) Do() (*StorePage, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// Update: Updates the content of a store page.
func (r *StorelayoutpagesService) Update(enterpriseId string, pageId string, storepage *StorePage) *StorelayoutpagesUpdateCall {
	c := &StorelayoutpagesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.pageId = pageId
	c.storepage = storepage
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StorelayoutpagesUpdateCall) QuotaUser(quotaUser string) *StorelayoutpagesUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StorelayoutpagesUpdateCall) UserIP(userIP string) *StorelayoutpagesUpdateCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *StorelayoutpagesUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.storepage)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/storeLayout/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"pageId":       c.pageId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.storelayoutpages.update" call.
// Exactly one of *StorePage or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StorePage.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StorelayoutpagesUpdateCall) Do() (*StorePage, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// method id "androidenterprise.users.generateToken":

type UsersGenerateTokenCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// GenerateToken: Generates a token (activation code) to allow this user
// to configure their work account in the Android Setup Wizard. Revokes
// any previously generated token.
func (r *UsersService) GenerateToken(enterpriseId string, userId string) *UsersGenerateTokenCall {
	c := &UsersGenerateTokenCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersGenerateTokenCall) QuotaUser(quotaUser string) *UsersGenerateTokenCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersGenerateTokenCall) UserIP(userIP string) *UsersGenerateTokenCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *UsersGenerateTokenCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/token")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.users.generateToken" call.
// Exactly one of *UserToken or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserToken.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersGenerateTokenCall) Do() (*UserToken, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generates a token (activation code) to allow this user to configure their work account in the Android Setup Wizard. Revokes any previously generated token.",
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
}

// Get: Retrieves a user's details.
func (r *UsersService) Get(enterpriseId string, userId string) *UsersGetCall {
	c := &UsersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersGetCall) QuotaUser(quotaUser string) *UsersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersGetCall) UserIP(userIP string) *UsersGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *UsersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.users.get" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *UsersGetCall) Do() (*User, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
}

// GetAvailableProductSet: Retrieves the set of products a user is
// entitled to access.
func (r *UsersService) GetAvailableProductSet(enterpriseId string, userId string) *UsersGetAvailableProductSetCall {
	c := &UsersGetAvailableProductSetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersGetAvailableProductSetCall) QuotaUser(quotaUser string) *UsersGetAvailableProductSetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersGetAvailableProductSetCall) UserIP(userIP string) *UsersGetAvailableProductSetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *UsersGetAvailableProductSetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/availableProductSet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.users.getAvailableProductSet" call.
// Exactly one of *ProductSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProductSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersGetAvailableProductSetCall) Do() (*ProductSet, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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

// method id "androidenterprise.users.list":

type UsersListCall struct {
	s            *Service
	enterpriseId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Looks up a user by email address. This only works for Google
// managed users.
func (r *UsersService) List(enterpriseId string, email string) *UsersListCall {
	c := &UsersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.urlParams_.Set("email", email)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersListCall) QuotaUser(quotaUser string) *UsersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersListCall) UserIP(userIP string) *UsersListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *UsersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.users.list" call.
// Exactly one of *UsersListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *UsersListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersListCall) Do() (*UsersListResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Looks up a user by email address. This only works for Google managed users.",
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

// method id "androidenterprise.users.revokeToken":

type UsersRevokeTokenCall struct {
	s            *Service
	enterpriseId string
	userId       string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// RevokeToken: Revokes a previously generated token (activation code)
// for the user.
func (r *UsersService) RevokeToken(enterpriseId string, userId string) *UsersRevokeTokenCall {
	c := &UsersRevokeTokenCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersRevokeTokenCall) QuotaUser(quotaUser string) *UsersRevokeTokenCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersRevokeTokenCall) UserIP(userIP string) *UsersRevokeTokenCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *UsersRevokeTokenCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/token")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.users.revokeToken" call.
func (c *UsersRevokeTokenCall) Do() error {
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
}

// SetAvailableProductSet: Modifies the set of products a user is
// entitled to access.
func (r *UsersService) SetAvailableProductSet(enterpriseId string, userId string, productset *ProductSet) *UsersSetAvailableProductSetCall {
	c := &UsersSetAvailableProductSetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterpriseId = enterpriseId
	c.userId = userId
	c.productset = productset
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersSetAvailableProductSetCall) QuotaUser(quotaUser string) *UsersSetAvailableProductSetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersSetAvailableProductSetCall) UserIP(userIP string) *UsersSetAvailableProductSetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *UsersSetAvailableProductSetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.productset)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "enterprises/{enterpriseId}/users/{userId}/availableProductSet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"enterpriseId": c.enterpriseId,
		"userId":       c.userId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "androidenterprise.users.setAvailableProductSet" call.
// Exactly one of *ProductSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ProductSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersSetAvailableProductSetCall) Do() (*ProductSet, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Modifies the set of products a user is entitled to access.",
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
